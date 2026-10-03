"""PPO training on SteppingEnv (push recovery, or walking w/ --gait-drive).

    .venv\\Scripts\\python.exe -m scripts.train_stepping_ppo --timesteps 1000000

--resume <zip> to continue (--timesteps is then *additional* steps). Use a
different --checkpoint-dir for each experiment.

SB3 gets imported inside main() on purpose - spawned workers re-import this
file and we don't want torch in every one of them.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# One BLAS thread per process (set before numpy import, also in workers).
for _thread_var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_thread_var, "1")

import numpy as np

from envs.stepping_env import SteppingEnv


# --assisted plant: 25 N*m human caps, AK80-64 in the abduction slot (the
# AK70-10 can't hold single-leg stance).
ASSISTED = dict(
    model_path="models/human_exo_direct_ak80abd.xml",
    calibration_path="models/exo_transmission_calibration_direct_ak80abd.json",
    exo_rated_torque_nm=(48.0, 48.0, 48.0, 48.0, 48.0, 48.0),
    joint_caps_nm=(25.0,) * 8,
)


def _gait_params(path):
    # {"params": [...]} json, or just a plain list
    if not path:
        return None
    d = json.loads(Path(path).read_text())
    return tuple(d["params"] if isinstance(d, dict) else d)


def env_kwargs(args) -> dict:
    # Shared by train + eval so they always build the same plant. The scorer
    # and viewer parsers don't have the two memory flags, hence the getattrs.
    kw = dict(control_decimation=args.decimation,
              arena_memory_mb=getattr(args, "arena_memory_mb", 16),
              discard_visual=not getattr(args, "keep_visuals", False),
              step_bonus=args.step_bonus,
              step_window_lo=args.step_window[0],
              step_window_hi=args.step_window[1],
              lock_ankle=not args.free_ankle,
              exo_action=args.exo_action,
              stabilize_foot=args.stabilize_foot,
              settle_decisions=args.settle_decisions,
              human_effort_weight=args.human_effort_weight,
              recovery_bonus=args.recovery_bonus,
              step_recovery_bonus=args.step_recovery_bonus,
              survival_bonus=args.survival_bonus,
              recovery_speed_bonus=args.recovery_speed_bonus,
              episode_seconds=args.episode_seconds,
              allocator_solver=args.allocator_solver,
              exo_decimation=args.exo_decimation,
              gait_drive=args.gait_drive,
              gait_params=_gait_params(args.gait_params),
              walk_progress_weight=args.walk_progress_weight,
              walk_target_speed=args.walk_target_speed,
              hip_adduction_cap_nm=args.hip_adduction_cap)
    if args.push_bodies:
        kw["push_bodies"] = tuple(args.push_bodies)
    if args.push_axes:
        kw["push_axes"] = tuple(args.push_axes)
    if args.push_omnidirectional:
        kw["push_omnidirectional"] = True
    if args.assisted:
        kw.update(ASSISTED)
        kw["human_offload"] = args.offload
        kw["hip_adduction_cap_nm"] = None
    # Must come after the ASSISTED update, which sets its own caps. Ankles keep
    # the base cap since the exo has no ankle motor.
    if args.joint_cap is not None:
        base = kw.get("joint_caps_nm", (25.0,) * 8)
        caps = [float(args.joint_cap)] * 8
        caps[3], caps[7] = base[3], base[7]
        kw["joint_caps_nm"] = tuple(caps)
    if args.exo_residual_scale:
        kw["exo_residual_scale"] = tuple(args.exo_residual_scale)
    # With --free-ankle, walking caps the ankles too (the gait was tuned that way).
    if args.ankle_cap is not None:
        caps = list(kw.get("joint_caps_nm", (25.0,) * 8))
        caps[3] = caps[7] = float(args.ankle_cap)
        kw["joint_caps_nm"] = tuple(caps)
    return kw


def make_env(rank: int, seed: int, push_lo: float, push_hi: float, kw: dict):
    # No per-env Monitor: VecMonitor wraps the vec env, and Monitor would pull
    # torch into each worker.
    def _init():
        return SteppingEnv(seed=seed + rank,
                           push_force_range=(push_lo, push_hi), **kw)
    return _init


def evaluate(model, episodes: int, push_lo: float, push_hi: float,
             kw: dict, seed0: int = 9000) -> dict[str, float]:
    # held-out seeds. model=None gives the zero-action baseline on the same
    # plant/forces/seeds
    env = SteppingEnv(push_force_range=(push_lo, push_hi), **kw)
    survived, lengths, margins = 0, [], []
    for i in range(episodes):
        obs, _ = env.reset(seed=seed0 + i)
        worst = float("inf")
        for step in range(4000):
            if model is None:
                action = np.zeros(env.action_space.shape, dtype=np.float32)
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            worst = min(worst, info["margin_m"])
            if term or trunc:
                break
        lengths.append(step + 1)
        margins.append(worst)
        survived += int(not info["fallen"])
    return {"survival_rate": survived / episodes,
            "mean_length": float(np.mean(lengths)),
            "mean_worst_margin_m": float(np.mean(margins))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--arena-memory-mb", type=int, default=16,
                    help="MiB per MuJoCo workspace; 0 keeps the original XML allocation")
    ap.add_argument("--keep-visuals", action="store_true",
                    help="retain render-only assets in rollout workers (uses more RAM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--push-force-range", type=float, nargs=2,
                    default=[60.0, 130.0], metavar=("MIN", "MAX"))
    ap.add_argument("--output", default="models/stepping_ppo")
    ap.add_argument("--checkpoint-dir", default="models/stepping_ppo_checkpoints")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--learning-rate", type=float, default=3e-4,
                    help="learning rate; use about 1e-4 when resuming")
    ap.add_argument("--eval-episodes", type=int, default=16)
    # Small net and simulation-bound, so CPU beats GPU here.
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--push-bodies", nargs="*", default=None,
                    help="restrict push targets, e.g. torso pelvis")
    ap.add_argument("--push-axes", type=int, nargs="*", default=None,
                    help="restrict push axes: 0 lateral, 1 sagittal")
    ap.add_argument("--push-omnidirectional", action="store_true",
                    help="push along a random horizontal heading instead of one world "
                         "axis; overrides --push-axes")
    ap.add_argument("--joint-cap", type=float, default=None,
                    help="human torque cap on the six non-ankle leg joints, N*m. "
                         "Lowering it is what shifts load onto the exo; ankle caps "
                         "keep the --assisted default")
    ap.add_argument("--ent-coef", type=float, default=0.005,
                    help="entropy coefficient; raise to ~0.03 if the policy needs to "
                         "explore large actions such as a step")
    ap.add_argument("--reset-log-std", type=float, default=None,
                    help="re-open exploration on resume, e.g. -0.7 for "
                         "sigma about 0.5")
    ap.add_argument("--step-window", type=float, nargs=2, default=(-0.15, -0.05),
                    metavar=("LO", "HI"),
                    help="XCoM margin window (m) in which --step-bonus is paid, "
                         "checked when one-foot support starts")
    ap.add_argument("--gamma", type=float, default=0.99,
                    help="discount factor. Use 0.99 for recovery (~200 decisions) and "
                         "0.997 for walking (~600). Ignored on resume")
    ap.add_argument("--target-kl", type=float, default=0.15,
                    help="stop the epoch loop once a minibatch exceeds this approx KL; "
                         "0 disables. SB3 stops at 1.5x this value")
    ap.add_argument("--step-bonus", type=float, default=0.0,
                    help="reward each time one-foot support starts while the XCoM "
                         "margin is inside --step-window")
    ap.add_argument("--decimation", type=int, default=20,
                    help="physics steps per policy decision")
    ap.add_argument("--vec", choices=("subproc", "sb3-subproc", "dummy"),
                    default="subproc",
                    help="vectorized env type. 'subproc' uses the lean workers that "
                         "skip importing torch; 'sb3-subproc' is the stock SB3 class")
    ap.add_argument("--free-ankle", action="store_true",
                    help="free the human's sagittal ankle (the exo still has no ankle "
                         "motor)")
    ap.add_argument("--exo-action", action="store_true",
                    help="policy commands a six-channel exo torque residual instead of "
                         "the human joint targets. Checkpoints do not carry over "
                         "between the two modes")
    ap.add_argument("--stabilize-foot", action="store_true",
                    help="PD-hold the subtalar joints near neutral (capped at 12 N*m)")
    ap.add_argument("--hip-adduction-cap", type=float, default=None,
                    help="raise only the human hip adduction cap, N*m")
    ap.add_argument("--assisted", action="store_true",
                    help="use the single-leg-stance plant: 25 N*m human cap, AK80-64 "
                         "hip abduction, offload 0.6")
    ap.add_argument("--offload", type=float, default=0.6,
                    help="human_offload used by --assisted")
    ap.add_argument("--settle-decisions", type=int, default=0,
                    help="decisions of quiet standing folded into the initial pose so "
                         "the feet are flat before the push")
    ap.add_argument("--human-effort-weight", type=float, default=0.05,
                    help="weight on the human-torque penalty. Raise it to push load "
                         "onto the device, but watch survival")
    ap.add_argument("--recovery-bonus", type=float, default=0.0,
                    help="reward per decision spent in a stable stance after the push")
    ap.add_argument("--allocator-solver", choices=("bvls", "pgd"), default="bvls",
                    help="least-squares backend for the exo allocation. 'bvls' (scipy) "
                         "is the default; 'pgd' is slower on CPU and only meant for "
                         "MJX parity checks")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="PPO minibatch size; keep it fixed across a paired "
                         "experiment. Omit on resume to keep the checkpoint value")
    ap.add_argument("--exo-decimation", type=int, default=1,
                    help="run the exo/measurement/safety stack every N physics steps "
                         "(human PD stays at 1 kHz). Changes the plant, so it must "
                         "match between train, score and view")
    ap.add_argument("--n-steps", type=int, default=None,
                    help="PPO rollout length per env. Keep envs x n_steps roughly "
                         "constant (~1536) when changing --envs; omit on resume to "
                         "keep the checkpoint value")
    ap.add_argument("--survival-bonus", type=float, default=1.0,
                    help="reward per decision for not having fallen; lower it (e.g. "
                         "0.5) to make settling worth more than staggering")
    ap.add_argument("--recovery-speed-bonus", type=float, default=0.0,
                    help="one-off bonus for recovering quickly, decaying linearly to "
                         "zero 150 decisions after the push ends")
    ap.add_argument("--step-recovery-bonus", type=float, default=0.0,
                    help="one-off reward when recovery follows a real step (foot rise "
                         "> 15 mm)")
    ap.add_argument("--exo-residual-scale", type=float, nargs=6, default=None,
                    help="per-motor bound on the exo torque residual, N*m, in order "
                         "abd_l abd_r flex_l flex_r knee_l knee_r")
    ap.add_argument("--ankle-cap", type=float, default=None,
                    help="human ankle torque cap, N*m, applied after --joint-cap; set "
                         "it equal to --joint-cap for walking")
    ap.add_argument("--gait-drive", action="store_true",
                    help="drive the human legs with a fixed periodic gait (walking). "
                         "Use with --exo-action, --gait-params and --push-force-range "
                         "0 0 (pushes are still applied otherwise)")
    ap.add_argument("--gait-params", default=None,
                    help="gait parameter JSON, e.g. models/gait_walk_base.json")
    ap.add_argument("--walk-target-speed", type=float, default=0.0,
                    help="forward speed (m/s) above which progress reward stops "
                         "paying; 0 disables. Keep it above the control arm's speed")
    ap.add_argument("--walk-progress-weight", type=float, default=0.0,
                    help="reward per m/s of forward CoM velocity, only used with "
                         "--gait-drive")
    ap.add_argument("--episode-seconds", type=float, default=4.0,
                    help="episode length in seconds")
    args = ap.parse_args()
    # Rollout overrides go into PPO.load so the buffer is built at the right
    # size; omitted ones keep the checkpoint's values.
    rollout_options = {key: getattr(args, key) for key in ("n_steps", "batch_size")
                       if getattr(args, key) is not None}
    if not args.resume:
        args.n_steps = 512 if args.n_steps is None else args.n_steps
        args.batch_size = 256 if args.batch_size is None else args.batch_size
    if (args.envs < 1 or args.timesteps < 0
            or (args.n_steps is not None and args.n_steps < 1)
            or (args.batch_size is not None and args.batch_size < 2)):
        ap.error("envs/n-steps must be positive, batch-size >= 2, timesteps >= 0")
    if args.n_steps is not None and args.envs * args.n_steps < 2:
        ap.error("PPO needs at least two samples per rollout")
    if args.arena_memory_mb < 0:
        ap.error("arena-memory-mb must be nonnegative (0 keeps XML settings)")

    # Imported here to keep torch out of the workers.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                                  VecMonitor)
    from envs.lean_subproc import LeanSubprocVecEnv

    lo, hi = args.push_force_range
    kw = env_kwargs(args)
    print(f"Simulation workspace: {args.arena_memory_mb or 'XML default'} MiB; "
          f"render assets {'retained' if args.keep_visuals else 'discarded'}")
    builders = [make_env(i, args.seed, lo, hi, kw) for i in range(args.envs)]
    # No fallback to DummyVecEnv if a spawn fails; let the run fail.
    if args.vec in ("subproc", "sb3-subproc"):
        cls = LeanSubprocVecEnv if args.vec == "subproc" else SubprocVecEnv
        vec = cls(builders)
    else:
        vec = DummyVecEnv(builders)
    vec = VecMonitor(vec)

    if args.resume:
        model = PPO.load(args.resume, env=vec, device=args.device,
                         **rollout_options)
        # The critic is fit to the old distribution; resume at a lower lr (~1e-4).
        model.learning_rate = args.learning_rate
        model._setup_lr_schedule()
        model.ent_coef = args.ent_coef
        model.target_kl = args.target_kl or None
        if args.reset_log_std is not None:
            # Re-open exploration after sigma has collapsed.
            import torch
            old_sigma = float(np.exp(model.policy.log_std.data.mean().item()))
            with torch.no_grad():
                model.policy.log_std.data.fill_(args.reset_log_std)
            print(f"reset log_std to {args.reset_log_std} "
                  f"(sigma {np.exp(args.reset_log_std):.3f})")
            # Also clear Adam state for the policy branch. The policy gradient
            # scales with 1/sigma^2, so after shrinking sigma the stale second
            # moments give huge steps. The critic's state is left alone.
            # The watchdog passes --reset-log-std on attempt 0 only, so restarts
            # don't clear it again.
            policy_branch = ("log_std", "action_net", "mlp_extractor.policy_net")
            cleared = [n for n, p in model.policy.named_parameters()
                       if n.startswith(policy_branch)
                       and model.policy.optimizer.state.pop(p, None) is not None]
            print(f"cleared Adam state for {len(cleared)} policy-branch tensors "
                  f"(sigma {old_sigma:.2f} -> {np.exp(args.reset_log_std):.3f}, "
                  f"gradient scale x{(old_sigma / np.exp(args.reset_log_std))**2:.0f})")
        print(f"resumed from {args.resume} at lr={args.learning_rate}, "
              f"ent_coef={args.ent_coef}")
    else:
        model = PPO(
            "MlpPolicy", vec, device=args.device, verbose=1,
            # gamma should match episode length: 0.99 (~100 decisions) for
            # recovery, 0.997 for 600-decision walking so a late fall is still
            # visible. Past ~0.999 the value function flattens.
            n_steps=args.n_steps, batch_size=args.batch_size, gae_lambda=0.95,
            gamma=args.gamma,
            learning_rate=args.learning_rate, ent_coef=args.ent_coef,
            clip_range=0.2, n_epochs=10, target_kl=args.target_kl or None,
            # Start with small sigma; unit variance knocks the model over
            # before it learns anything.
            policy_kwargs={"net_arch": [256, 256], "log_std_init": -1.5},
        )

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    callback = CheckpointCallback(save_freq=max(25000 // args.envs, 1),
                                  save_path=args.checkpoint_dir,
                                  name_prefix="stepping_ppo")
    try:
        if model.n_steps * args.envs < 2:
            raise ValueError("PPO needs at least two samples per rollout")
        print(f"PPO rollout: {model.n_steps} steps x {args.envs} envs = "
              f"{model.n_steps * args.envs} samples; batch_size={model.batch_size}; "
              f"gamma={model.gamma}")
        model.learn(total_timesteps=args.timesteps, callback=callback,
                    reset_num_timesteps=args.resume is None)
        model.save(args.output)
    finally:
        vec.close()
    print(f"saved {args.output}.zip")

    if args.eval_episodes <= 0:
        print("--eval-episodes 0: skipping the held-out evaluation")
        return 0

    stats = evaluate(model, args.eval_episodes, lo, hi, kw)
    print(f"held-out evaluation over {args.eval_episodes} episodes: "
          f"survival {100.0 * stats['survival_rate']:.0f}%, "
          f"mean length {stats['mean_length']:.0f} decisions, "
          f"mean worst margin {stats['mean_worst_margin_m']:+.3f} m")
    # Matched zero-action control.
    base = evaluate(None, args.eval_episodes, lo, hi, kw)
    print(f"zero-action baseline, same plant/band/seeds: "
          f"survival {100.0 * base['survival_rate']:.0f}%, "
          f"mean length {base['mean_length']:.0f} decisions, "
          f"mean worst margin {base['mean_worst_margin_m']:+.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
