"""Safety supervisor. Any hard limit latches an emergency, and while it's
latched the env drops the trunk hold. Not learned - the policy can't get
around these checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.fastmath import norm
from controllers.measurements import StandingMeasurement


@dataclass(frozen=True)
class SafetyConfig:
    max_root_error_rad: float = 0.35
    max_root_angular_speed_rad_s: float = 2.0
    cuff_soft_limit_n: float = 100.0
    cuff_hard_limit_n: float = 150.0
    minimum_contacts: int = 2
    stable_window_s: float = 0.25
    max_com_speed_m_s: float = 2.0
    startup_grace_s: float = 2.0


@dataclass(frozen=True)
class SafetyStatus:
    root_error_rad: float
    root_angular_speed_rad_s: float
    foot_contacts: int
    cuff_force_n: float
    com_speed_m_s: float
    emergency: bool
    stable: bool


class SafetySupervisor:
    """Any hard limit (nan state, root error, root ang vel, not enough foot
    contacts, cuff force, com speed) latches an emergency. Everything except
    the nan check is ignored for the first startup_grace_s. Unlatches after
    stable_window_s of stable samples in a row.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 root_reference: np.ndarray,
                 config: SafetyConfig | None = None) -> None:
        self.model = model
        self.data = data
        self.config = config or SafetyConfig()
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        if root < 0 or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("safety supervision requires a free root joint")
        self._root_vadr = int(model.jnt_dofadr[root])
        self.root_reference = np.asarray(root_reference, dtype=float).copy()
        # Time between read() calls, used by the stable-window clock.
        # SteppingEnv sets k * timestep when exo_decimation=k.
        self.period_s = float(model.opt.timestep)
        self._stable_time = 0.0
        self.emergency_latched = False

    def read(self, measurement: StandingMeasurement, cuff_force_n: float,
             update: bool = True) -> SafetyStatus:
        # update=False only reports: it leaves the latch and the stable-window
        # clock alone, so a state that gets read twice isn't counted twice.
        error = np.zeros(3)
        mujoco.mju_subQuat(error, self.root_reference, measurement.root_orientation)
        root_error = norm(error)
        angular_speed = norm(self.data.qvel[self._root_vadr + 3:self._root_vadr + 6])
        com_speed = norm(measurement.com_velocity)
        cfg = self.config
        active = measurement.time >= cfg.startup_grace_s
        hard = (not np.isfinite(self.data.qpos).all()
                or not np.isfinite(self.data.qvel).all()
                or (active and root_error > cfg.max_root_error_rad)
                or (active and angular_speed > cfg.max_root_angular_speed_rad_s)
                or (active and measurement.foot_contact_count < cfg.minimum_contacts)
                or (active and cuff_force_n > cfg.cuff_hard_limit_n)
                or (active and com_speed > cfg.max_com_speed_m_s))
        stable_sample = (root_error < 0.12 and angular_speed < 0.45
                         and measurement.foot_contact_count >= cfg.minimum_contacts
                         and cuff_force_n < cfg.cuff_soft_limit_n
                         and com_speed < 0.75)
        if update:
            if hard:
                self.emergency_latched = True
                self._stable_time = 0.0
            if stable_sample and not hard:
                self._stable_time += self.period_s
                if self._stable_time >= cfg.stable_window_s:
                    self.emergency_latched = False
            else:
                self._stable_time = 0.0
        return SafetyStatus(root_error, angular_speed,
                            measurement.foot_contact_count, float(cuff_force_n),
                            com_speed, self.emergency_latched,
                            self._stable_time >= cfg.stable_window_s)

    def reset(self) -> None:
        self._stable_time = 0.0
        self.emergency_latched = False
