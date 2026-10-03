"""Hard-coded human controller (stand-in for the wearer's own motor control).

Writes human torques to qfrc_applied only, never data.ctrl (exo_assistance
owns the motors).

Trunk is held upright by a reaction-less PD torque on the root free joint
(capped at `max_root_torque_nm`). It is dropped when the caller passes
`allow_root_recovery=False`, which the env does while a safety emergency is
latched.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.fastmath import clip


HUMAN_STANDING_JOINTS = (
    "hip_flexion_r", "hip_adduction_r", "knee_angle_r", "ankle_angle_r",
    "hip_flexion_l", "hip_adduction_l", "knee_angle_l", "ankle_angle_l",
)

# Ankle entries in HUMAN_STANDING_JOINTS order.
_ANKLE_INDICES = np.array([3, 7])

# Subtalar (inversion/eversion) is otherwise uncommanded: the muscles spanning
# it get no ctrl, so without this the foot rolls onto its edge under lateral
# pushes. Toes (mtp) are left out; they would need their own, smaller cap.
FOOT_STABILIZER_JOINTS = ("subtalar_angle_r", "subtalar_angle_l")


@dataclass(frozen=True)
class HumanBaselineConfig:
    kp: float = 500.0
    kd: float = 25.0
    root_kp: float = 500.0
    root_kd: float = 25.0
    ankle_kp: float = 500.0
    ankle_kd: float = 25.0
    hold_root_orientation: bool = True
    max_joint_torque_nm: float = 25.0
    # Per-joint caps in HUMAN_STANDING_JOINTS order; None uses
    # `max_joint_torque_nm` everywhere. Hip adduction usually needs more: full
    # single stance with feet at x = +-0.084 m takes ~70 N*m (m*g*d).
    joint_torque_caps_nm: tuple[float, ...] | None = None
    max_root_torque_nm: float = 100.0
    # Subtalar stabiliser (see FOOT_STABILIZER_JOINTS). Off by default. 12 N*m
    # is about a quarter of the model's own subtalar muscle capacity.
    stabilize_foot: bool = False
    foot_kp: float = 150.0
    foot_kd: float = 5.0
    max_foot_torque_nm: float = 12.0


class HumanBaselineController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config: HumanBaselineConfig | None = None) -> None:
        self.model = model
        self.data = data
        self.config = config or HumanBaselineConfig()

        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in HUMAN_STANDING_JOINTS
        ]
        missing = [name for name, jid in zip(HUMAN_STANDING_JOINTS, joint_ids)
                   if jid < 0]
        if missing:
            raise ValueError(f"human standing joints missing: {missing}")
        self._qadr = np.asarray([model.jnt_qposadr[j] for j in joint_ids], dtype=int)
        self._vadr = np.asarray([model.jnt_dofadr[j] for j in joint_ids], dtype=int)

        caps = self.config.joint_torque_caps_nm
        if caps is None:
            self._caps = np.full(len(HUMAN_STANDING_JOINTS),
                                 float(self.config.max_joint_torque_nm))
        else:
            self._caps = np.asarray(caps, dtype=float)
            if self._caps.shape != (len(HUMAN_STANDING_JOINTS),):
                raise ValueError(
                    "joint_torque_caps_nm must contain one entry per joint in "
                    f"HUMAN_STANDING_JOINTS: {HUMAN_STANDING_JOINTS}")

        self._foot_qadr = np.zeros(0, dtype=int)
        self._foot_vadr = np.zeros(0, dtype=int)
        if self.config.stabilize_foot:
            foot_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
                        for n in FOOT_STABILIZER_JOINTS]
            missing = [n for n, j in zip(FOOT_STABILIZER_JOINTS, foot_ids) if j < 0]
            if missing:
                raise ValueError(f"foot stabiliser joints missing: {missing}")
            self._foot_qadr = np.asarray([model.jnt_qposadr[j] for j in foot_ids],
                                         dtype=int)
            self._foot_vadr = np.asarray([model.jnt_dofadr[j] for j in foot_ids],
                                         dtype=int)

        self.last_joint_torque = np.zeros(len(HUMAN_STANDING_JOINTS))
        self.last_unscaled_joint_torque = np.zeros(len(HUMAN_STANDING_JOINTS))
        self.last_root_torque = np.zeros(3)
        self._root_quat: np.ndarray | None = None
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root >= 0 and model.jnt_type[root] == mujoco.mjtJoint.mjJNT_FREE:
            self._root_vadr = int(model.jnt_dofadr[root])
            self._root_qadr = int(model.jnt_qposadr[root])
            self._root_quat = data.qpos[self._root_qadr + 3:self._root_qadr + 7].copy()

    @property
    def joint_names(self) -> tuple[str, ...]:
        return HUMAN_STANDING_JOINTS

    def apply(self, joint_targets: np.ndarray | None = None,
              allow_root_recovery: bool = True) -> None:
        # last_unscaled_joint_torque is the PD torque before the per-joint cap;
        # the env hands a share of it to the exo when offloading.
        cfg = self.config
        if joint_targets is None:
            joint_targets = np.zeros(len(HUMAN_STANDING_JOINTS))
        # Copy: the ankle entries are zeroed below and must not reach the
        # caller's buffer.
        joint_targets = np.array(joint_targets, dtype=float)
        if joint_targets.shape != (len(HUMAN_STANDING_JOINTS),):
            raise ValueError("joint_targets must contain eight human leg-joint entries")
        # Ankles are always held at neutral.
        joint_targets[_ANKLE_INDICES] = 0.0
        self.last_unscaled_joint_torque = (
            cfg.kp * (joint_targets - self.data.qpos[self._qadr])
            + cfg.kd * (0.0 - self.data.qvel[self._vadr])
        )
        self.last_unscaled_joint_torque[_ANKLE_INDICES] = (
            cfg.ankle_kp * (joint_targets[_ANKLE_INDICES]
                            - self.data.qpos[self._qadr[_ANKLE_INDICES]])
            - cfg.ankle_kd * self.data.qvel[self._vadr[_ANKLE_INDICES]]
        )
        self.last_joint_torque = clip(
            self.last_unscaled_joint_torque, -self._caps, self._caps)
        self.data.qfrc_applied[self._vadr] = self.last_joint_torque

        if self._foot_qadr.size:
            # Subtalar hold near neutral. Never offloaded or policy-commanded;
            # the exo has no ankle or foot actuator.
            self.data.qfrc_applied[self._foot_vadr] = clip(
                cfg.foot_kp * (0.0 - self.data.qpos[self._foot_qadr])
                - cfg.foot_kd * self.data.qvel[self._foot_vadr],
                -cfg.max_foot_torque_nm, cfg.max_foot_torque_nm)

        if allow_root_recovery and cfg.hold_root_orientation and self._root_quat is not None:
            # Reaction-less trunk hold: torque goes straight onto the root free
            # joint with no equal and opposite reaction anywhere in the body.
            error = np.zeros(3)
            mujoco.mju_subQuat(
                error,
                self._root_quat,
                self.data.qpos[self._root_qadr + 3:self._root_qadr + 7],
            )
            angular_vadr = self._root_vadr + 3
            self.last_root_torque = (
                cfg.root_kp * error - cfg.root_kd * self.data.qvel[angular_vadr:angular_vadr + 3]
            )
            self.last_root_torque = clip(
                self.last_root_torque, -cfg.max_root_torque_nm, cfg.max_root_torque_nm
            )
            self.data.qfrc_applied[angular_vadr:angular_vadr + 3] = self.last_root_torque
        elif not allow_root_recovery:
            self.last_root_torque[:] = 0.0
