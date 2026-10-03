"""Torso balance demand (root orientation error -> hip/knee torque)."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.fastmath import clip_scalar


@dataclass(frozen=True)
class TorsoBalanceConfig:
    kp: float = 250.0
    kd: float = 32.0
    # Exo-equivalent torque limit, not the human's own joint cap.
    max_joint_torque_nm: float = 100.0


class TorsoBalanceController:
    # Hip/knee strategy only since there's no ankle motor. The allocator maps
    # this through the exo->human coupling afterwards.

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 root_reference: np.ndarray,
                 config: TorsoBalanceConfig | None = None) -> None:
        self.model = model
        self.data = data
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_vadr = int(model.jnt_dofadr[root])
        self.root_qadr = int(model.jnt_qposadr[root])
        self.root_reference = np.asarray(root_reference, dtype=float).copy()
        self.config = config or TorsoBalanceConfig()

    def compute(self) -> np.ndarray:
        """Returns the 8-entry human-equivalent demand."""
        error = np.zeros(3)
        mujoco.mju_subQuat(
            error, self.root_reference,
            self.data.qpos[self.root_qadr + 3:self.root_qadr + 7],
        )
        omega = self.data.qvel[self.root_vadr + 3:self.root_vadr + 6]
        root_torque = self.config.kp * error - self.config.kd * omega

        # The root reference is rotated -90 deg about z, so error channel 0 is
        # lateral and 1 is sagittal. Yaw is left to contact.
        limit = 2.0 * self.config.max_joint_torque_nm
        sagittal = clip_scalar(root_torque[1], -limit, limit)
        lateral = clip_scalar(root_torque[0], -limit, limit)
        demand = np.zeros(8)
        demand[[0, 2, 4, 6]] = 0.25 * sagittal
        demand[1] = 0.50 * lateral
        demand[5] = -0.50 * lateral
        return demand
