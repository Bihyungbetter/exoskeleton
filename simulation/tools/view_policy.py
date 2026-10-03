"""Watch a walking policy (or the control) in the MuJoCo viewer.

Env is built the same way as in evaluate_walking.py so you're looking at the
same plant that got scored. Use the same flags (see
runs/run_view_walking_decim5.cmd).

    --policy PATH   a trained policy (deterministic mean action)
    --policy none   the control: the fixed exo law with a zero residual
"""
from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from envs.stepping_env import SteppingEnv
from tools.evaluate_walking import add_env_flags, walking_env_kwargs

# Strap and cuff-shell visuals. They have no geom group, so hide them by name.
CUFF_VISUAL_GEOMS = (
    "strap_waist", "strap_thigh_r", "strap_thigh_l",
    "strap_shank_r", "strap_shank_l",
    "exo_cuff_shell_waist", "exo_cuff_shell_thigh_r", "exo_cuff_shell_thigh_l",
    "exo_cuff_shell_shank_r", "exo_cuff_shell_shank_l",
)


def hide_cuff_visuals(model) -> None:
    for name in CUFF_VISUAL_GEOMS:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            model.geom_rgba[gid, 3] = 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_flags(ap)
    ap.add_argument("--policy", default="models/walking_decim5_20260916.zip")
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--loop", action="store_true",
                    help="replay the episodes until the window is closed")
    ap.add_argument("--realtime", type=float, default=1.0,
                    help="playback speed; 0 runs as fast as possible")
    ap.add_argument("--hide-cuffs", action="store_true")
    ap.add_argument("--telemetry-every", type=int, default=0,
                    help="print exo and human torques every N decisions")
    args = ap.parse_args()

    kw = walking_env_kwargs(args)
    kw["human_offload"] = args.offload
    # Training drops visual geoms to save memory; the viewer keeps them.
    # Trajectories are identical either way.
    kw["discard_visual"] = False
    env = SteppingEnv(**kw)
    if args.hide_cuffs:
        hide_cuff_visuals(env.model)

    policy = None
    if args.policy.lower() != "none":
        from stable_baselines3 import PPO
        policy = PPO.load(args.policy, device="cpu")
    print("policy:", args.policy if policy else "NONE (control: zero residual)")
    if args.telemetry_every:
        print("exo = [hipAbd_l hipAbd_r hipFlex_l hipFlex_r knee_l knee_r] N*m, "
              "human = [hipFlex_r hipAdd_r knee_r ankle_r hipFlex_l hipAdd_l "
              "knee_l ankle_l] N*m")
    print(f"{'seed':>6} {'result':>8} {'secs':>6} {'travel':>8} {'exoPk':>7} {'humanPk':>8}")

    dt = env.control_decimation * env.model.opt.timestep
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        ep = 0
        while viewer.is_running():
            if ep >= args.episodes:
                if not args.loop:
                    break
                ep = 0
            seed = args.seed0 + ep
            ep += 1
            # The viewer renders `data` on its own thread, so hold the lock.
            with viewer.lock():
                obs, _ = env.reset(seed=seed)
            viewer.sync()
            n, exo_pk, human_pk, info, ended = 0, 0.0, 0.0, {}, False
            while viewer.is_running():
                wall = time.time()
                if policy is None:
                    action = np.zeros(env.action_space.shape, dtype=np.float32)
                else:
                    action, _ = policy.predict(obs, deterministic=True)
                with viewer.lock():
                    obs, _r, term, trunc, info = env.step(action)
                viewer.sync()
                n += 1
                exo_pk = max(exo_pk, float(np.abs(info["exo_command"]).max()))
                human_pk = max(human_pk, float(np.abs(info["human_joint_torque"]).max()))
                if args.telemetry_every and n % args.telemetry_every == 0:
                    e = " ".join(f"{v:6.1f}" for v in info["exo_command"])
                    h = " ".join(f"{v:6.1f}" for v in info["human_joint_torque"])
                    print(f"  dec {n:4d}  exo=[{e}]  human=[{h}]")
                if term or trunc:
                    ended = True
                    break
                if args.realtime > 0:
                    left = dt / args.realtime - (time.time() - wall)
                    if left > 0:
                        time.sleep(left)
            if info:
                result = ("fell" if info.get("fallen") else
                          "walked" if ended else "stopped")
                print(f"{seed:6d} {result:>8} {n * dt:6.1f} "
                      f"{float(info.get('travel_y', 0.0)):7.2f}m "
                      f"{exo_pk:6.1f}Nm {human_pk:7.1f}Nm", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
