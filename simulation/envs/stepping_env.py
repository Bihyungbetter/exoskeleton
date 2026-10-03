"""Gym env for push recovery / walking with the hip+knee exo.

Default: policy outputs targets for the human's 6 non-ankle joints (stepping
reflex), exo runs its fixed law.
exo_action=True: policy outputs a bounded 6-ch exo torque residual instead.
gait_drive=True: human follows a fixed periodic gait (walking). Pushes still
happen unless push_force_range=(0, 0), which is what the walking scripts use.

The safety limits are NOT learned - exo keeps rated torque/slew/cuff limits
and human torque stays capped (25 N*m default).
"""
from __future__ import annotations

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from controllers.balance_margin import compute_balance_margin
from controllers.exo_assistance import (EXO_ACTUATORS, ExoAssistanceConfig,
                                        ExoAssistanceController)
from controllers.fastmath import clip_scalar
from controllers.human_baseline import HumanBaselineConfig, HumanBaselineController
from controllers.measurements import StandingMeasurements
from controllers.safety import SafetyConfig, SafetySupervisor
from controllers.torso_balance import TorsoBalanceConfig, TorsoBalanceController
from controllers.wholebody_allocator import WholeBodyAllocator
from envs.plant_setup import align_exo_to_human, drop_onto_floor, lock_ankles
from tools.model_paths import check_arena_capacity, load_model

# Leg joints the reflex commands. No ankles: there's no ankle actuator and the
# ankles are locked, so an ankle command would just be a reward hack.
ACTION_JOINTS = ("hip_flexion_r", "hip_adduction_r", "knee_angle_r",
                 "hip_flexion_l", "hip_adduction_l", "knee_angle_l")
ACTION_INDEX = (0, 1, 2, 4, 5, 6)   # into the eight-entry human joint vector
# Per-joint command range (rad), about what a recovery step needs.
ACTION_SCALE = np.array([0.60, 0.35, 0.90, 0.60, 0.35, 0.90])
# Stable decisions needed to count as recovered (0.5 s at 50 Hz).
_RECOVERY_DWELL = 25
# A step = contacts gone AND foot risen this much. Looser counts contact chatter.
_STEP_CLEARANCE_M = 0.015
# Margin below which a step answers a real emergency. One foot alone puts the
# margin near -0.05 m, so a looser value is satisfied by the lift itself.
_STEP_EMERGENCY_MARGIN_M = -0.15
# The step-recovery bonus only pays if recovery follows within this many
# decisions. Also the decay length of the recovery-speed bonus.
_STEP_RECOVERY_WINDOW = 150


class SteppingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, model_path: str = "models/human_exo_direct.xml",
                 calibration_path: str = "models/exo_transmission_calibration_direct.json",
                 episode_seconds: float = 4.0,
                 seed: int | None = None,
                 push_force_range: tuple[float, float] = (60.0, 130.0),
                 push_duration_steps: int = 200,
                 cuff_hard_limit_n: float = 150.0,
                 human_offload: float = 0.0,
                 control_decimation: int = 20,
                 push_bodies: tuple[str, ...] = ("torso", "pelvis",
                                                 "femur_r", "tibia_r"),
                 push_axes: tuple[int, ...] = (0, 1),
                 push_omnidirectional: bool = False,
                 step_bonus: float = 0.0,
                 step_window_lo: float = -0.15,
                 step_window_hi: float = -0.05,
                 lock_ankle: bool = True,
                 exo_action: bool = False,
                 exo_residual_scale: tuple = (5.0, 5.0, 20.0, 20.0, 20.0, 20.0),
                 action_penalty_weight: float | None = None,
                 stabilize_foot: bool = False,
                 gait_drive: bool = False,
                 gait_params: tuple | None = None,
                 walk_progress_weight: float = 0.0,
                 walk_target_speed: float = 0.0,
                 hip_adduction_cap_nm: float | None = None,
                 joint_caps_nm: tuple[float, ...] | None = None,
                 exo_rated_torque_nm: tuple[float, ...] | None = None,
                 settle_decisions: int = 0,
                 human_effort_weight: float = 0.05,
                 recovery_bonus: float = 0.0,
                 step_recovery_bonus: float = 0.0,
                 survival_bonus: float = 1.0,
                 recovery_speed_bonus: float = 0.0,
                 root_orientation_hold: bool = True,
                 allocator_solver: str = "bvls",
                 exo_decimation: int = 1,
                 arena_memory_mb: int = 16,
                 discard_visual: bool = False) -> None:
        super().__init__()
        self.model_path = model_path
        self.arena_memory_mb = arena_memory_mb
        self.discard_visual = discard_visual
        self.calibration_path = calibration_path
        self.episode_seconds = episode_seconds
        self.push_force_range = (float(push_force_range[0]),
                                 float(push_force_range[1]))
        self.push_duration_steps = int(push_duration_steps)
        self.cuff_hard_limit_n = float(cuff_hard_limit_n)
        self.human_offload = float(human_offload)
        # Physics steps per policy decision. 20 -> 50 Hz on the 1 kHz plant;
        # acting every physics step puts consequences beyond the discount horizon.
        self.control_decimation = max(1, int(control_decimation))
        # Push geometry dominates survival, so keep it controllable
        # (e.g. torso/pelvis on axis 0 for the lateral case).
        self.push_bodies = tuple(push_bodies)
        self.push_axes = tuple(push_axes)
        # Random horizontal heading instead of +-one axis. Superset of the axis
        # pushes, same observation size.
        self.push_omnidirectional = bool(push_omnidirectional)
        # Bonus for lifting a foot, only inside the danger window below.
        self.step_bonus = float(step_bonus)
        # Margin window (m) in which the step bonus pays; see the gate in step().
        self.step_window_lo = float(step_window_lo)
        self.step_window_hi = float(step_window_hi)
        # Locking the ankles also removes the human's ankle strategy, not just
        # the (nonexistent) exo ankle motor. Default True for comparability.
        self.lock_ankle = bool(lock_ankle)
        # False: action = six human joint targets, exo on its fixed law.
        # True: action = normalized exo torque residual on top of the fixed
        # allocation; the human holds neutral targets.
        # Channel order differs between modes: ACTION_JOINTS is grouped by side
        # (flex_r, add_r, knee_r, flex_l, add_l, knee_l), EXO_ACTUATORS by joint
        # (abd_l, abd_r, flex_l, flex_r, knee_l, knee_r). Checkpoints don't
        # transfer between modes.
        self.exo_action = bool(exo_action)
        # N*m per unit residual, in EXO_ACTUATORS order. Abduction is smaller
        # because the AK70-10 is only 8.3 N*m continuous.
        self.exo_residual_scale = np.asarray(exo_residual_scale, dtype=float)
        # Action-magnitude penalty. Off by default in exo mode, where it would
        # become an exo-torque penalty.
        self.action_penalty_weight = float(
            action_penalty_weight if action_penalty_weight is not None
            else (0.0 if exo_action else 0.02))
        # Hold the two subtalar joints near neutral (otherwise they hit their
        # limit in lateral topples).
        self.stabilize_foot = bool(stabilize_foot)
        # Fixed periodic gait as the human-side driver (walking). Pair with
        # exo_action=True so the policy only commands the exo motors.
        # gait_params is the six-entry GaitParams vector; the dataclass
        # defaults don't walk, so pass a tuned one (models/gait_walk_base.json).
        self.gait_drive = bool(gait_drive)
        self.gait_params = (None if gait_params is None
                            else tuple(float(x) for x in gait_params))
        # Reward per m/s of CoM velocity along world y (sagittal). Walking only.
        # Needs to dominate, otherwise standing still minimizes effort.
        self.walk_progress_weight = float(walk_progress_weight)
        # Speed cap (m/s) on the progress reward; 0 = uncapped. Uncapped, the
        # policy learns to throw the wearer forward and fall. Backwards still costs.
        self.walk_target_speed = float(walk_target_speed)
        # Override only the hip adduction cap. Single support needs ~70 N*m of
        # stance-hip abduction (85.5 kg * g * 0.084 m), above the 25 N*m default.
        self.hip_adduction_cap_nm = hip_adduction_cap_nm
        # Full eight-entry cap override, ordered like HUMAN_STANDING_JOINTS.
        self.joint_caps_nm = (tuple(float(c) for c in joint_caps_nm)
                              if joint_caps_nm is not None else None)
        # Rated continuous exo torque in N*m. The config stores fractions of
        # ctrlrange, which change meaning if a model variant resizes an actuator.
        self.exo_rated_torque_nm = (tuple(float(t) for t in exo_rated_torque_nm)
                                    if exo_rated_torque_nm is not None else None)
        # Quiet-standing decisions folded into the initial pose so episodes start
        # feet-flat (see _settle_to_flat_feet). 120 is enough; 0 = off.
        self.settle_decisions = int(settle_decisions)
        # Weight on normalized human effort. There is intentionally no exo-torque
        # reward: in exo mode it'd be farmable with opposing torques that cancel.
        # The term maxes at this weight vs +1 per decision, so 0.05 is mild.
        self.human_effort_weight = float(human_effort_weight)
        # Per-decision reward for stable stance after the push. Survival alone
        # pays equally for staggering until the time limit.
        self.recovery_bonus = float(recovery_bonus)
        # One-off reward on first recovery, only if a real step happened first.
        # A plain step bonus mostly pays for the leg coming up during a topple.
        self.step_recovery_bonus = float(step_recovery_bonus)
        self.survival_bonus = float(survival_bonus)
        self.recovery_speed_bonus = float(recovery_speed_bonus)
        # Reaction-less root orientation hold in the human baseline (up to
        # 100 N*m). Keeps the trunk square, which also blocks the lateral lean
        # single-leg stance needs.
        self.root_orientation_hold = bool(root_orientation_hold)
        # Allocator backend, runs every exo step. "bvls" = SciPy (reference),
        # "pgd" = fixed-iteration projected gradient.
        self.allocator_solver = str(allocator_solver)
        # Run exo/measurement/safety every N physics steps; human PD, push and
        # mj_step stay at 1 kHz and ctrl holds in between. This changes the
        # plant: anything other than 1 needs its own baseline.
        self.exo_decimation = int(exo_decimation)
        if (self.exo_decimation < 1
                or self.control_decimation % self.exo_decimation != 0):
            raise ValueError("exo_decimation must be a positive divisor of "
                             "control_decimation so the controller period "
                             "matches its safety and slew-limit clock")
        self._rng = np.random.default_rng(seed)
        self.action_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        # 3 root error, 3 root ang vel, 3 CoM pos, 3 CoM vel, 2 XCoM, 1 margin,
        # 2 GRF, 2 contacts, 8 human q, 8 human dq, 6 last action = 41.
        # Keep in sync with _observation (it checks the shape).
        obs_dim = 41
        if self.gait_drive:
            obs_dim += 4      # sin phase, cos phase, swing side, cuff force
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,),
                                            dtype=np.float32)
        self._build()

    def _torque_caps(self) -> tuple[float, ...] | None:
        # None -> uniform 25 N*m
        if self.joint_caps_nm is not None:
            if len(self.joint_caps_nm) != 8:
                raise ValueError("joint_caps_nm needs one entry per joint in "
                                 "HUMAN_STANDING_JOINTS")
            return self.joint_caps_nm
        if self.hip_adduction_cap_nm is None:
            return None
        # HUMAN_STANDING_JOINTS order: hip_flex, hip_add, knee, ankle; right then left.
        cap = float(self.hip_adduction_cap_nm)
        return (25.0, cap, 25.0, 25.0, 25.0, cap, 25.0, 25.0)

    def _build(self) -> None:
        self.model = load_model(self.model_path, arena_memory_mb=self.arena_memory_mb,
                                discard_visual=self.discard_visual)
        if self.lock_ankle:
            lock_ankles(self.model)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        drop_onto_floor(self.model, self.data)
        align_exo_to_human(self.model, self.data)
        check_arena_capacity(self.data)
        self._initial_qpos = self.data.qpos.copy()

        self.measurements = StandingMeasurements(self.model)
        caps = self._torque_caps()
        self.human = HumanBaselineController(
            self.model, self.data,
            HumanBaselineConfig(max_joint_torque_nm=25.0,
                                stabilize_foot=self.stabilize_foot,
                                hold_root_orientation=self.root_orientation_hold,
                                joint_torque_caps_nm=caps))
        # Normalize effort by the caps actually applied, so the term is cap-invariant.
        self._effort_norm = (
            np.asarray([caps[i] for i in ACTION_INDEX], dtype=float)
            if caps is not None else np.full(len(ACTION_INDEX), 25.0))
        exo_config = dict(max_cuff_force_n=0.66 * self.cuff_hard_limit_n,
                          hard_cuff_force_n=self.cuff_hard_limit_n,
                          cuff_governor_knee_n=0.53 * self.cuff_hard_limit_n,
                          max_torque_fraction=1.0)
        if self.exo_rated_torque_nm is not None:
            peak = np.array([
                self.model.actuator_ctrlrange[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n), 1]
                for n in EXO_ACTUATORS])
            rated = np.asarray(self.exo_rated_torque_nm, dtype=float)
            if rated.shape != peak.shape:
                raise ValueError("exo_rated_torque_nm needs one entry per exo "
                                 f"actuator: {EXO_ACTUATORS}")
            if np.any(rated > peak):
                raise ValueError(
                    "rated torque exceeds the model's peak ctrlrange "
                    f"{dict(zip(EXO_ACTUATORS, peak))}; resize the actuator in "
                    "the model rather than raising the rated limit past it")
            exo_config["rated_torque_fraction"] = tuple(rated / peak)
        self.exo = ExoAssistanceController(
            self.model, self.data, ExoAssistanceConfig(**exo_config))
        self.exo.control_period_s = (self.exo_decimation
                                     * float(self.model.opt.timestep))
        self.allocator = WholeBodyAllocator(self.model, self.calibration_path,
                                            solver=self.allocator_solver)
        root = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qadr = int(self.model.jnt_qposadr[root])
        self.root_vadr = int(self.model.jnt_dofadr[root])
        self.root_reference = self.data.qpos[
            self.root_qadr + 3:self.root_qadr + 7].copy()
        self.balance = TorsoBalanceController(
            self.model, self.data, self.root_reference, TorsoBalanceConfig())
        # Support/posture limits are looser than the defaults since stepping
        # spends time on one foot. The cuff and nonfinite checks aren't.
        self.safety = SafetySupervisor(
            self.model, self.data, self.root_reference,
            SafetyConfig(startup_grace_s=0.5,
                         cuff_soft_limit_n=0.66 * self.cuff_hard_limit_n,
                         cuff_hard_limit_n=self.cuff_hard_limit_n,
                         minimum_contacts=1, max_root_error_rad=0.85,
                         max_root_angular_speed_rad_s=4.0))
        self.safety.period_s = self.exo.control_period_s
        # Last supervisor status, reused between exo updates when exo_decimation > 1.
        self._last_status = None
        self._last_cuff_n = 0.0
        self._foot_body = {side: mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, f"calcn_{side}")
            for side in ("r", "l")}
        self._human_qadr = np.array([
            self.model.jnt_qposadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.human.joint_names])
        self._human_vadr = np.array([
            self.model.jnt_dofadr[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in self.human.joint_names])
        self._time_limit_steps = int(round(
            self.episode_seconds / self.model.opt.timestep))
        self._time_limit_decisions = max(
            1, self._time_limit_steps // self.control_decimation)
        self._last_action = np.zeros(6)
        self._steps = 0
        self._decisions = 0
        self._nominal_com_height = float(self.data.subtree_com[0][2])
        self._stable_decisions = 0
        self._stepped = False
        self._step_reward_paid = False
        # Rising-edge state for the step bonus; also reset in reset().
        self._prev_one_foot = False
        self._recovery_speed_paid = False
        self._step_decision = 0
        self._prev_margin_m = 0.0
        self._foot_base_z = {k: float(self.data.xpos[b][2])
                             for k, b in self._foot_body.items()}
        if self.settle_decisions > 0:
            self._settle_to_flat_feet(self.settle_decisions)
        # Lazy import keeps the default path's import closure unchanged.
        self._gait = None
        if self.gait_drive:
            from controllers.gait import GaitController, GaitParams
            self._gait = GaitController(
                GaitParams() if self.gait_params is None
                else GaitParams.from_vector(np.asarray(self.gait_params)))

    def _settle_to_flat_feet(self, decisions: int) -> None:
        # After drop_onto_floor only the heels touch (neutral pose is a bit
        # toes-up) so each foot only gives ~20mm of ML base instead of ~65mm.
        # Standing quietly for ~100 decisions gets the feet flat; we save that
        # pose as the start pose. Only done once per env.
        saved_push = (getattr(self, "_push_body", 0), getattr(self, "_push_axis", 0),
                      getattr(self, "_push_force", 0.0),
                      getattr(self, "_push_xy", None))
        self._push_body, self._push_axis, self._push_force = 0, 0, 0.0
        # _advance reads _push_xy and no heading has been drawn yet.
        self._push_xy = None
        self._push_start, self._push_end = 1 << 30, 1 << 30
        self.exo.reset()
        self.safety.reset()
        for _ in range(decisions):
            self._advance(np.zeros(8))
        self._initial_qpos = self.data.qpos.copy()
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self._steps = 0
        self._decisions = 0
        self.exo.reset()
        self.safety.reset()
        (self._push_body, self._push_axis, self._push_force,
         self._push_xy) = saved_push
        mujoco.mj_forward(self.model, self.data)

    def _advance(self, targets: np.ndarray,
                 exo_residual: np.ndarray | None = None
                 ) -> tuple[bool, np.ndarray]:
        """One decision's worth of physics steps. Used by step() and by the
        settling so they run identical dynamics.

        exo_residual is in EXO_ACTUATORS order, None = plain exo law.
        Returns (terminated_early, last exo command).
        """
        terminated_early = False
        command = self.exo.last_command.copy()
        k_exo = self.exo_decimation
        for k in range(self.control_decimation):
            self.data.qfrc_applied[:] = 0.0
            self.data.xfrc_applied[:] = 0.0
            if self._push_start <= self._steps < self._push_end:
                if self._push_xy is None:
                    self.data.xfrc_applied[self._push_body, self._push_axis] = \
                        self._push_force
                else:
                    self.data.xfrc_applied[self._push_body, 0:2] = self._push_xy

            exo_step = (k % k_exo == 0)
            if exo_step:
                m_inner = self.measurements.read(self.data)
                # One cuff read shared by supervisor, governor and projection.
                cuff = self.exo.cuff_force_peak()
                status = self.safety.read(m_inner, cuff)
                self._last_status = status
            else:
                status = self._last_status
            self.human.apply(joint_targets=targets,
                             allow_root_recovery=not status.emergency)

            if exo_step:
                # Stored for the observation (residual_gain depends on it).
                self._last_cuff_n = float(cuff)
                assist_ramp = clip_scalar((self.data.time - 0.5) / 0.5, 0.0, 1.0)
                desired = assist_ramp * 0.05 * self.balance.compute()
                # Offload: the exo is also asked for a share of the human's
                # uncapped PD torque. The human still applies its own capped
                # torque, so this adds to it rather than replacing it.
                if self.human_offload > 0.0:
                    offload = (assist_ramp * self.human_offload
                               * self.human.last_unscaled_joint_torque)
                    offload[[3, 7]] = 0.0
                    desired = desired + offload
                allocated = self.allocator.allocate(
                    self.model, self.data, desired,
                    rated_limit=self.exo.rated_limit(),
                    last_command=self.exo.last_command,
                    slew_limit=500.0 * self.exo.control_period_s,
                    damping=2.0, cuff_gain=self.exo.cuff_governor_gain(cuff))
                if exo_residual is not None:
                    # Policy residual on top of the fixed allocation, faded out
                    # between 80 and 120 N of cuff force. apply_allocated still
                    # does the final limit projection.
                    residual_gain = clip_scalar((120.0 - cuff) / 40.0, 0.0, 1.0)
                    allocated = allocated + (assist_ramp * residual_gain
                                             * exo_residual
                                             * self.exo_residual_scale)
                command = self.exo.apply_allocated(allocated, cuff)

            mujoco.mj_step(self.model, self.data)
            self._steps += 1
            # Stop early on a fall so no reward accrues for physics after it.
            if not np.isfinite(self.data.qpos).all():
                terminated_early = True
                break
            if float(self.data.subtree_com[0][2]) < 0.65 * self._nominal_com_height:
                terminated_early = True
                break
        check_arena_capacity(self.data)
        check_arena_capacity(self.allocator._scratch)
        return terminated_early, command

    def _detect_touchdown(self, m) -> None:
        # Same touchdown detector the gait was tuned with. Foot has to actually
        # be in the air first (no contacts + above _STEP_CLEARANCE_M). On a
        # timeout swap we reset since the swing leg changed under us.
        swing = self._gait.swing
        contacts = m.right_contacts if swing == "r" else m.left_contacts
        rise = (float(self.data.xpos[self._foot_body[swing]][2])
                - self._foot_base_z[swing])
        if self._gait_timed_out:
            self._gait_prev_air, self._gait_cleared = False, False
        air = contacts == 0 and rise > _STEP_CLEARANCE_M
        self._gait_cleared = self._gait_cleared or air
        if self._gait_prev_air and not air and self._gait_cleared:
            self._gait_touchdowns += 1
            self._gait_cleared = False
            self._gait.on_touchdown()
        self._gait_prev_air = air

    def _observation(self, m, margin) -> np.ndarray:
        error = np.zeros(3)
        # Reference is rotated -90 deg about z: channel 0 is lateral, 1 sagittal.
        mujoco.mju_subQuat(error, self.root_reference, m.root_orientation)
        obs = np.concatenate([
            error,
            self.data.qvel[self.root_vadr + 3:self.root_vadr + 6],
            m.com_position, m.com_velocity,
            margin.xcom,
            [margin.margin_m if np.isfinite(margin.margin_m) else -1.0],
            [m.right_grf_n / 500.0, m.left_grf_n / 500.0],
            [float(m.right_contacts > 0), float(m.left_contacts > 0)],
            self.data.qpos[self._human_qadr], self.data.qvel[self._human_vadr],
            self._last_action,
        ])
        if self._gait is not None:
            # Gait phase as sin/cos (no wrap discontinuity) plus swing side as a
            # label. Without phase, assistance can only be a posture bias.
            ph = 2.0 * np.pi * float(self._gait.phase)
            obs = np.concatenate([obs, [
                np.sin(ph), np.cos(ph),
                1.0 if self._gait.swing == "r" else -1.0,
                # Cuff force / hard limit, so the policy can see the residual
                # fade in _advance (walking already sits partly inside it).
                float(self._last_cuff_n) / max(self.cuff_hard_limit_n, 1e-9),
            ]])
        obs = obs.astype(np.float32)
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(
                "observation is %s but observation_space is %s; a channel was "
                "added to _observation without updating obs_dim"
                % (obs.shape, self.observation_space.shape))
        return obs

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self.data.qpos[:] = self._initial_qpos
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0
        self.data.ctrl[:] = 0.0
        self.data.time = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.exo.reset()
        self.safety.reset()
        self._steps = 0
        self._decisions = 0
        self._last_action[:] = 0.0
        self._stable_decisions = 0
        self._stepped = False
        self._step_reward_paid = False
        # Must reset here too, or a foot left airborne suppresses the next
        # episode's first step.
        self._prev_one_foot = False
        self._recovery_speed_paid = False
        self._step_decision = 0
        self._prev_margin_m = 0.0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if self._gait is not None:
            self._gait.reset()
            self._gait_prev_air = False
            self._gait_cleared = False
            self._gait_touchdowns = 0
        # Don't reorder the RNG draws below: it changes the seed -> push mapping.
        body_names = self.push_bodies
        self._push_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY,
            body_names[int(self._rng.integers(0, len(body_names)))])
        # Horizontal axes only; vertical pushes barely topple anything.
        self._push_axis = int(
            self.push_axes[int(self._rng.integers(0, len(self.push_axes)))])
        direction = -1.0 if self._rng.random() < 0.5 else 1.0
        self._push_force = direction * float(
            self._rng.uniform(*self.push_force_range))
        self._push_start = int(self._rng.integers(700, 1400))
        self._push_end = self._push_start + self.push_duration_steps
        self.data.qvel[self.root_vadr + 3:self.root_vadr + 6] = \
            self._rng.normal(0, 0.03, 3)
        # Heading is drawn last and only when enabled, so axis-aligned runs
        # consume the same randomness as before.
        self._push_xy = None
        if self.push_omnidirectional:
            theta = float(self._rng.uniform(0.0, 2.0 * np.pi))
            magnitude = abs(float(self._push_force))
            self._push_xy = magnitude * np.array([np.cos(theta), np.sin(theta)])
        mujoco.mj_forward(self.model, self.data)
        check_arena_capacity(self.data)
        # Foot baseline must be read after mj_forward; xpos is stale until then.
        self._foot_base_z = {k: float(self.data.xpos[b][2])
                             for k, b in self._foot_body.items()}
        m = self.measurements.read(self.data)
        margin = compute_balance_margin(m)
        if self._gait is not None:
            self._walk_y0 = float(m.com_position[1])
            self._walk_prev_y = self._walk_y0
        return self._observation(m, margin), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        # Action penalty only sees what the policy commanded.
        policy_action = action.copy()
        # Human targets: the policy's action in default mode, neutral in exo
        # mode. The two action spaces are never summed.
        human_action = np.zeros(6) if self.exo_action else action
        if self._gait is not None:
            # Gait is added on top (in exo mode it's the whole human command).
            # Leg swap comes from the real touchdown, not the clock.
            self._gait_timed_out = self._gait.advance(
                self.control_decimation * self.model.opt.timestep)
            human_action = np.clip(human_action + self._gait.action(),
                                   -1.0, 1.0)
        # Offsets from neutral, held for the whole window.
        targets = np.zeros(8)
        targets[list(ACTION_INDEX)] = human_action * ACTION_SCALE
        exo_residual = policy_action if self.exo_action else None

        terminated_early, command = self._advance(targets, exo_residual)
        self._decisions += 1
        # Last action in the obs = whatever the policy itself controls.
        self._last_action = (policy_action.copy() if self.exo_action
                             else human_action.copy())

        m = self.measurements.read(self.data)
        if self._gait is not None:
            self._detect_touchdown(m)
        margin = compute_balance_margin(m)
        # The next decision's first exo tick reads this same state and updates
        # the supervisor then; reading it here too would count it twice.
        new_status = self.safety.read(m, self.exo.cuff_force_peak(), update=False)

        # Recovery step: one foot off and clear, after push start, while the
        # previous decision's margin was below the emergency threshold. Uses the
        # previous margin because the lift alone drives the margin negative.
        # Latched per episode.
        emergency = (np.isfinite(self._prev_margin_m)
                     and self._prev_margin_m < _STEP_EMERGENCY_MARGIN_M)
        if self._steps >= self._push_start and emergency:
            for side, body in self._foot_body.items():
                contacts = m.right_contacts if side == "r" else m.left_contacts
                if (contacts == 0 and float(self.data.xpos[body][2])
                        - self._foot_base_z[side] > _STEP_CLEARANCE_M):
                    if not self._stepped:
                        self._step_decision = self._decisions
                    self._stepped = True

        # "Recovered" = stable for _RECOVERY_DWELL decisions after the push:
        # both feet loaded, CoM high enough, XCoM inside support, CoM nearly
        # still, trunk near upright. Thresholds are hand-picked.
        post_push = self._steps > self._push_end
        stable = bool(
            post_push
            # Load, not contact count: a grazing contact would pass as double support.
            # 100 N is ~12% of body weight.
            and m.right_grf_n > 100.0 and m.left_grf_n > 100.0
            # Catches a buckled stance leg that passes everything else.
            and float(m.com_position[2]) > 0.90 * self._nominal_com_height
            and np.isfinite(margin.margin_m) and margin.margin_m > 0.0
            and float(np.linalg.norm(m.com_velocity[:2])) < 0.10
            and float(new_status.root_error_rad) < 0.20)
        self._stable_decisions = self._stable_decisions + 1 if stable else 0
        recovered = self._stable_decisions >= _RECOVERY_DWELL

        com_height = float(m.com_position[2])
        settled = self._steps >= 200
        fallen = bool(
            com_height < 0.65 * self._nominal_com_height
            or new_status.root_error_rad > 0.8
            or (settled and m.foot_contact_count < 1))
        terminated = (fallen or terminated_early
                      or not np.all(np.isfinite(self.data.qpos)))
        truncated = self._decisions >= self._time_limit_decisions

        margin_m = margin.margin_m if np.isfinite(margin.margin_m) else -0.5
        deficit = max(0.0, -margin_m)
        # Mean squared human torque over the six non-ankle joints (ankles are
        # locked), normalized by the actual caps.
        human_effort = float(np.mean(
            (self.human.last_joint_torque[list(ACTION_INDEX)]
             / self._effort_norm) ** 2))
        step_reward = 0.0
        if self.step_bonus > 0.0:
            airborne = int(m.right_contacts == 0) + int(m.left_contacts == 0)
            one_foot = airborne == 1 and m.foot_contact_count >= 1
            # Pay only when the previous margin is in the window where the
            # outcome is still in doubt. Above it nearly everything recovers
            # anyway; below -0.15 nothing does, so no gradient.
            danger = (np.isfinite(self._prev_margin_m)
                      and self.step_window_hi > self._prev_margin_m
                      > self.step_window_lo)
            # Rising edge only, so hovering on one foot doesn't keep paying.
            if one_foot and danger and not self._prev_one_foot:
                step_reward = self.step_bonus
            self._prev_one_foot = one_foot
        if self._gait is not None:
            # Walking reward. Separate from the recovery reward, whose margin
            # penalty would tax every step and whose push terms don't apply.
            # Progress is a per-decision delta along world y (sagittal), so it
            # telescopes to distance. Effort is on the six joints the exo can
            # act on. Cuff and action penalties as in recovery.
            progress_m = float(m.com_position[1]) - self._walk_prev_y
            self._walk_prev_y = float(m.com_position[1])
            speed = progress_m / (self.control_decimation
                                  * self.model.opt.timestep)
            if self.walk_target_speed > 0.0:
                speed = min(speed, self.walk_target_speed)
            reward = (
                self.walk_progress_weight * speed
                + self.survival_bonus
                - 0.5 * float(new_status.root_error_rad) ** 2
                - self.human_effort_weight * human_effort
                - self.action_penalty_weight * float(np.mean(policy_action ** 2))
                - 0.30 * max(0.0, new_status.cuff_force_n - 100.0) / 50.0
            )
            if terminated:
                reward -= 20.0
            info = {"safety": new_status.__dict__, "margin_m": margin_m,
                    "recovered": False, "stable_decisions": self._stable_decisions,
                    "stepped": self._gait_touchdowns > 0,
                    "exo_command": command.copy(),
                    "human_joint_torque": self.human.last_joint_torque.copy(),
                    "root_hold_torque": self.human.last_root_torque.copy(),
                    "com_height": com_height, "fallen": fallen,
                    "human_effort": human_effort,
                    "travel_y": float(m.com_position[1]) - self._walk_y0,
                    "touchdowns": self._gait_touchdowns}
            self._prev_margin_m = margin_m
            return (self._observation(m, margin), float(reward), terminated,
                    truncated, info)
        reward = (
            # Survival term; configurable since it saturates at easy push forces.
            self.survival_bonus
            - 3.0 * min(deficit, 0.5) ** 2
            - 0.5 * float(new_status.root_error_rad) ** 2
            - self.human_effort_weight * human_effort
            - self.action_penalty_weight * float(np.mean(policy_action ** 2))
            - 0.30 * max(0.0, new_status.cuff_force_n - 100.0) / 50.0
            + step_reward
            + self.recovery_bonus * float(stable)
        )
        if (recovered and self._stepped and not self._step_reward_paid
                and self._decisions - self._step_decision <= _STEP_RECOVERY_WINDOW
                and self.step_recovery_bonus > 0.0):
            reward += self.step_recovery_bonus
            self._step_reward_paid = True
        # One-off bonus for recovering quickly: full at push end, linearly to
        # zero _STEP_RECOVERY_WINDOW decisions later.
        if (recovered and not self._recovery_speed_paid
                and self.recovery_speed_bonus > 0.0):
            since = self._decisions - self._push_end / self.control_decimation
            promptness = float(np.clip(
                1.0 - since / _STEP_RECOVERY_WINDOW, 0.0, 1.0))
            reward += self.recovery_speed_bonus * promptness
            self._recovery_speed_paid = True
        if terminated:
            # Roughly the survival reward forfeited by falling early.
            reward -= 20.0

        info = {"safety": new_status.__dict__, "margin_m": margin_m,
                "recovered": recovered, "stable_decisions": self._stable_decisions,
                "stepped": self._stepped,
                "exo_command": command.copy(),
                "human_joint_torque": self.human.last_joint_torque.copy(),
                "com_height": com_height, "fallen": fallen}
        # Next decision's step gates use this pre-lift margin.
        self._prev_margin_m = margin_m
        return self._observation(m, margin), float(reward), terminated, truncated, info
