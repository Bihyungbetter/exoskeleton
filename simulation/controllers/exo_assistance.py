"""Exo motor controller. Apart from the env zeroing ctrl on reset, nothing else
writes data.ctrl (the human side uses qfrc_applied). Every command goes
through _project: rated torque clip, slew limit, cuff force trip. A trip cuts
torque to zero straight away, without the slew limit.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from controllers.fastmath import clip, clip_scalar, norm


EXO_ACTUATORS = (
    "exo_act_hip_abd_l", "exo_act_hip_abd_r", "exo_act_hip_flex_l",
    "exo_act_hip_flex_r", "exo_act_knee_l", "exo_act_knee_r",
)
EXO_JOINTS = (
    "exo_hip_abd_l", "exo_hip_abd_r", "exo_hip_flex_l",
    "exo_hip_flex_r", "exo_knee_l", "exo_knee_r",
)


@dataclass(frozen=True)
class ExoAssistanceConfig:
    # torques in N*m, forces in N
    max_torque_fraction: float = 0.25
    max_torque_rate_nm_s: float = 500.0
    max_cuff_force_n: float = 100.0
    hard_cuff_force_n: float = 150.0
    # Predictive cuff governor: extrapolate the cuff force trend by the horizon
    # and derate task torque linearly from the knee down to zero at
    # max_cuff_force_n, before the hard trip.
    cuff_predict_horizon_s: float = 0.02
    cuff_governor_knee_n: float = 80.0
    # Rated/peak per motor, in EXO_ACTUATORS order: AK70-10 abduction
    # (8.3/24.8 N*m), AK80-64 hip flexion and knee (48/120 N*m).
    rated_torque_fraction: tuple[float, ...] = (
        8.3 / 24.8, 8.3 / 24.8, 48.0 / 120.0, 48.0 / 120.0,
        48.0 / 120.0, 48.0 / 120.0,
    )


class ExoAssistanceController:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 config: ExoAssistanceConfig | None = None) -> None:
        self.model = model
        self.data = data
        self.config = config or ExoAssistanceConfig()
        self._actuator_ids = self._ids(mujoco.mjtObj.mjOBJ_ACTUATOR, EXO_ACTUATORS)
        self._peak = model.actuator_ctrlrange[self._actuator_ids, 1].copy()
        # Seconds between control updates, used by the slew limit and the cuff
        # trend. `SteppingEnv(exo_decimation=k)` sets it to k * timestep.
        self.control_period_s = float(model.opt.timestep)
        self._rated_limit = self._compute_rated_limit()
        self.last_command = np.zeros(len(EXO_ACTUATORS))
        self.safety_trip = False
        self._previous_cuff_force = 0.0
        self._cuff_force_sensor_ids = [
            sid for sid in range(model.nsensor)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or "").startswith("cuff_")
            and (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, sid) or "").endswith("_force")
        ]
        self._cuff_force_slices = [
            slice(int(model.sensor_adr[sid]), int(model.sensor_adr[sid]) + 3)
            for sid in self._cuff_force_sensor_ids]

    def _ids(self, obj: mujoco.mjtObj, names: tuple[str, ...]) -> np.ndarray:
        ids = np.asarray([mujoco.mj_name2id(self.model, obj, name) for name in names], dtype=int)
        if np.any(ids < 0):
            missing = [name for name, idx in zip(names, ids) if idx < 0]
            raise ValueError(f"missing exoskeleton model names: {missing}")
        return ids

    def rated_limit(self) -> np.ndarray:
        return self._rated_limit.copy()

    def _compute_rated_limit(self) -> np.ndarray:
        cfg = self.config
        rated_fraction = np.asarray(cfg.rated_torque_fraction, dtype=float)
        if rated_fraction.shape != (len(EXO_ACTUATORS),):
            raise ValueError("rated_torque_fraction must contain six entries")
        return np.minimum(cfg.max_torque_fraction * self._peak,
                          rated_fraction * self._peak)

    def apply_allocated(self, command: np.ndarray,
                        cuff_force_n: float | None = None) -> np.ndarray:
        """Send a command from WholeBodyAllocator through _project.

        The allocator already respects the limits, but projecting again keeps
        this class the final authority on ``data.ctrl``. ``cuff_force_n`` is the
        peak cuff force if the caller already read it; None reads the sensors.
        """
        command = np.asarray(command, dtype=float)
        if command.shape != (len(EXO_ACTUATORS),):
            raise ValueError("command must have one entry per exo actuator")
        if cuff_force_n is None:
            cuff_force_n = self.cuff_force_peak()
        if self.safety_trip:
            # Stay off until cuff load is back under the soft limit; the slew
            # limit then ramps torque back in.
            if cuff_force_n > self.config.max_cuff_force_n:
                self.disable()
                return self.last_command.copy()
            self.safety_trip = False
        return self._project(command, cuff_force_n)

    def _project(self, command: np.ndarray, cuff_force_n: float) -> np.ndarray:
        cfg = self.config
        limit = self._rated_limit
        command = clip(command, -limit, limit)
        if not np.isfinite(command).all():
            self.safety_trip = True
            self.disable()
            return self.last_command.copy()
        if cfg.max_torque_rate_nm_s > 0.0:
            max_delta = cfg.max_torque_rate_nm_s * self.control_period_s
            command = clip(command, self.last_command - max_delta,
                           self.last_command + max_delta)
        if cuff_force_n > cfg.hard_cuff_force_n > 0.0:
            self.safety_trip = True
            self.disable()
            return self.last_command.copy()
        self.data.ctrl[self._actuator_ids] = command
        self.last_command = command.copy()
        return command.copy()

    def cuff_governor_gain(self, cuff_force_n: float | None = None) -> float:
        # 0..1 scale on the task torque. This updates the stored cuff trend, so
        # only call it once per control step (before the allocator solve).
        # Only a rising cuff force is extrapolated, a falling one shouldn't buy
        # extra headroom.
        cfg = self.config
        cuff_now = (self.cuff_force_peak() if cuff_force_n is None
                    else float(cuff_force_n))
        rate = (cuff_now - self._previous_cuff_force) / max(self.control_period_s, 1e-9)
        self._previous_cuff_force = cuff_now
        predicted = cuff_now + max(0.0, rate) * cfg.cuff_predict_horizon_s
        span = cfg.max_cuff_force_n - cfg.cuff_governor_knee_n
        if span <= 0.0:
            return 1.0 if predicted <= cfg.max_cuff_force_n else 0.0
        return clip_scalar((cfg.max_cuff_force_n - predicted) / span, 0.0, 1.0)

    def reset(self) -> None:
        self.last_command[:] = 0.0
        self.safety_trip = False
        self._previous_cuff_force = 0.0

    def disable(self) -> None:
        self.data.ctrl[self._actuator_ids] = 0.0
        self.last_command[:] = 0.0

    def cuff_force_peak(self) -> float:
        peak = 0.0
        sensordata = self.data.sensordata
        for sl in self._cuff_force_slices:
            peak = max(peak, norm(sensordata[sl]))
        return peak
