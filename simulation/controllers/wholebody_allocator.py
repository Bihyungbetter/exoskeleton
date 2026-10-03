"""Box-constrained exo torque allocation with gravity support as a fixed offset.

Gravity support comes from inverse dynamics, not a posture PD. A PD has to
deviate from its setpoint to produce torque, so it can't be gravity
compensation; it just adds stiffness that fights the balance command.

The task torque is solved as bounded least squares around the support offset.
Rated torque, slew and cuff budget all go into the task bounds, so the task
only gets what's left after support (during fast changes the slew limit can
still cut into the support itself). Clipping per actuator after an
unconstrained solve would lose the solved direction, since the transmission is
cross-coupled.

The (8, 6) transmission M has an effective rank of about 4 (singular values
down to 0.0074), which limits how much human torque the exo can relieve.
"""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import lsq_linear

from controllers.exo_assistance import EXO_JOINTS
from controllers.fastmath import clip


def projected_gradient(a: np.ndarray, b: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray, iters: int = 50) -> np.ndarray:
    """Accelerated projected gradient for box-constrained least squares.

    Fixed iterations, no branches, so it can be jitted for an MJX port later.
    It's ~13x slower than BVLS on CPU so only use it for that. Matches BVLS at
    50 iterations.
    """
    ata = a.T @ a
    atb = a.T @ b
    lipschitz = float(np.linalg.eigvalsh(ata)[-1])
    step = 1.0 / max(lipschitz, 1e-12)
    x = np.clip(0.5 * (lo + hi), lo, hi)
    y = x.copy()
    t = 1.0
    for _ in range(iters):
        grad = ata @ y - atb
        x_new = np.clip(y - step * grad, lo, hi)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = np.clip(x_new + ((t - 1.0) / t_new) * (x_new - x), lo, hi)
        x, t = x_new, t_new
    return x


class WholeBodyAllocator:
    """solver: "bvls" (scipy lsq_linear) or "pgd" (projected_gradient above).
    They don't give bit-identical results, so switching = a plant change.
    """

    def __init__(self, model: mujoco.MjModel, calibration_path: str,
                 regularization: float = 1e-3,
                 solver: str = "bvls", pgd_iters: int = 50) -> None:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        self.matrix = np.asarray(
            calibration["human_constraint_torque_per_exo_torque"], dtype=float)
        if self.matrix.shape != (8, 6):
            raise ValueError("calibration matrix must have shape (8, 6)")
        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                     for n in EXO_JOINTS]
        if any(j < 0 for j in joint_ids):
            raise ValueError("missing exoskeleton joint names")
        self._vadr = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)
        if solver not in ("bvls", "pgd"):
            raise ValueError(f"solver must be 'bvls' or 'pgd', got {solver!r}")
        self.solver = solver
        self.pgd_iters = int(pgd_iters)
        # Scratch MjData so the dynamics evaluations don't touch live state.
        self._scratch = mujoco.MjData(model)
        self._rne = np.zeros(model.nv)
        # Solve matrix [M; sqrt(reg) I]. The Tikhonov rows keep the solve
        # well-posed in M's near-null directions.
        self._a = np.vstack([self.matrix, np.sqrt(regularization) * np.eye(6)])

    def gravity_support(self, model: mujoco.MjModel,
                        data: mujoco.MjData) -> np.ndarray:
        # Gravity only (qfrc_bias minus coriolis). Just running the pieces
        # of mj_forward we need on the scratch data, same answer, way cheaper.
        self._scratch.qpos[:] = data.qpos
        self._scratch.qvel[:] = 0.0
        self._scratch.ctrl[:] = 0.0
        self._scratch.act[:] = 0.0
        mujoco.mj_kinematics(model, self._scratch)
        mujoco.mj_comPos(model, self._scratch)
        mujoco.mj_comVel(model, self._scratch)
        mujoco.mj_rne(model, self._scratch, 0, self._rne)
        return self._rne[self._vadr].copy()

    def allocate(self, model: mujoco.MjModel, data: mujoco.MjData,
                 desired_human_torque: np.ndarray,
                 rated_limit: np.ndarray,
                 last_command: np.ndarray,
                 slew_limit: float,
                 damping: float = 0.0,
                 cuff_gain: float = 1.0) -> np.ndarray:
        # desired_human_torque: 8 entries, balance demand + any relief.
        # cuff_gain only shrinks the task box, never the support.
        desired = np.asarray(desired_human_torque, dtype=float)
        if desired.shape != (8,):
            raise ValueError("desired_human_torque must contain eight entries")
        rated_limit = np.asarray(rated_limit, dtype=float)
        last_command = np.asarray(last_command, dtype=float)

        support = self.gravity_support(model, data)
        if damping > 0.0:
            # Damping only, no position hold, so it can't pull the joint back
            # to neutral against a recovery command.
            support = support - damping * data.qvel[self._vadr]
        # Clip support to the motor; a saturated axis leaves the task nothing.
        support = clip(support, -rated_limit, rated_limit)

        # Bounds on the task torque, after support.
        lo = -rated_limit - support
        hi = rated_limit - support
        if slew_limit > 0.0:
            lo = np.maximum(lo, last_command - slew_limit - support)
            hi = np.minimum(hi, last_command + slew_limit - support)
        if cuff_gain < 1.0:
            # Shrink toward zero task torque, not toward the support offset.
            span = np.maximum(np.abs(lo), np.abs(hi)) * cuff_gain
            lo = np.maximum(lo, -span)
            hi = np.minimum(hi, span)
        # The box can collapse (support at the limit, or cuff gain 0), and
        # lsq_linear needs lo < hi, so open it by eps around the midpoint.
        mid = 0.5 * (lo + hi)
        eps = 1e-9
        lo = np.minimum(lo, mid - eps)
        hi = np.maximum(hi, mid + eps)

        # Support already acts on the human joints through M; solve for the rest.
        residual = desired - self.matrix @ support
        b = np.concatenate([residual, np.zeros(6)])
        if self.solver == "pgd":
            task = projected_gradient(self._a, b, lo, hi, iters=self.pgd_iters)
        else:
            task = lsq_linear(self._a, b, bounds=(lo, hi), method="bvls").x
        return support + task
