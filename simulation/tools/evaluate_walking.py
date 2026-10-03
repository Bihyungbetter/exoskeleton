"""Compare walking arms: distance, falls, and how much leg torque the exo takes.

The env comes from train_stepping_ppo.env_kwargs, so pass the same plant flags
as the trainer (including --exo-decimation). Load is averaged over the window
of the shortest run across all arms, and arms are paired per seed against the
first arm.

Arms:

    zero            no policy residual, --offload as given
    offload:F       no policy residual, human_offload = F
    model:PATH[:F]  trained exo policy, optionally with its own offload

With no policy the exo still runs its fixed law (gravity support, damping and
a small balance term), so offload:0 is a nearly passive device rather than no
device.

runs\\score_walking_decim5.cmd has the full command behind the README numbers
(48 seeds from 40000).
"""
from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import numpy as np

from envs.stepping_env import SteppingEnv
from scripts.train_stepping_ppo import env_kwargs

ANKLE_INDEX = (3, 7)
LEG_INDEX = (0, 1, 2, 4, 5, 6)


def parse_arm(spec: str, default_offload: float) -> dict:
    if spec == "zero":
        return dict(name=spec, model=None, offload=default_offload)
    if spec.startswith("offload:"):
        return dict(name=spec, model=None, offload=float(spec.split(":", 1)[1]))
    if spec.startswith("model:"):
        rest = spec[len("model:"):]
        # Split from the right so a Windows path keeps its drive letter.
        if ":" in rest and rest.rsplit(":", 1)[1].replace(".", "").isdigit():
            path, off = rest.rsplit(":", 1)
            return dict(name=spec, model=path, offload=float(off))
        return dict(name=spec, model=rest, offload=default_offload)
    raise ValueError("unrecognized arm %r" % spec)


def rollout(env: SteppingEnv, model, seed: int, max_decisions: int) -> dict:
    obs, _ = env.reset(seed=seed)
    human, exo, cuff, root, travel = [], [], [], [], []
    fell, info = True, {}
    n = 0
    for i in range(max_decisions):
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)
        n = i + 1
        human.append(np.abs(info["human_joint_torque"]))
        exo.append(np.abs(info["exo_command"]))
        cuff.append(float(info["safety"]["cuff_force_n"]))
        root.append(float(np.abs(info["root_hold_torque"]).max()))
        travel.append(float(info["travel_y"]))
        if term or trunc:
            fell = bool(info.get("fallen", True))
            break
    else:
        fell = False
    dt = env.control_decimation * env.model.opt.timestep
    return dict(decisions=n, seconds=n * dt, fell=fell,
                travel_y=float(info.get("travel_y", 0.0)),
                touchdowns=int(info.get("touchdowns", 0)),
                human=np.asarray(human), exo=np.asarray(exo),
                cuff=np.asarray(cuff), root=np.asarray(root),
                travel_series=np.asarray(travel))


def score(runs: list, window: int) -> dict:
    per_seed_human = np.array([r["human"][:window][:, list(LEG_INDEX)].mean()
                               for r in runs])
    per_seed_exo = np.array([r["exo"][:window].mean() for r in runs])
    ankle = np.array([r["human"][:window][:, list(ANKLE_INDEX)].mean()
                      for r in runs])
    root = np.array([r["root"][:window].mean() for r in runs])
    root_sat = np.array([(r["root"][:window] >= 99.9).mean() for r in runs])
    human_leg = float(per_seed_human.mean())
    exo_mean = float(per_seed_exo.mean())
    return dict(
        human_leg_nm=human_leg,
        human_leg_sd=float(per_seed_human.std(ddof=1)) if len(runs) > 1 else 0.0,
        human_ankle_nm=float(ankle.mean()),
        exo_nm=exo_mean,
        # Ankle excluded: the exo has no ankle motor.
        exo_share=exo_mean / max(exo_mean + human_leg, 1e-9),
        root_hold_nm=float(root.mean()),
        root_sat_frac=float(root_sat.mean()),
        cuff_peak_n=float(max(r["cuff"][:window].max() for r in runs)),
        travel_y=float(np.mean([r["travel_y"] for r in runs])),
        # Distance inside the common window. Whole-episode travel_y also
        # counts the forward fall.
        travel_window_m=float(np.mean([r["travel_series"][window - 1]
                                       - r["travel_series"][0]
                                       for r in runs])),
        travel_sd=float(np.std([r["travel_y"] for r in runs], ddof=1))
        if len(runs) > 1 else 0.0,
        seconds=float(np.mean([r["seconds"] for r in runs])),
        touchdowns=float(np.mean([r["touchdowns"] for r in runs])),
        fell=int(sum(r["fell"] for r in runs)),
        per_seed_human=per_seed_human.tolist(),
        per_seed_exo=per_seed_exo.tolist(),
        per_seed_travel=[float(r["travel_series"][window - 1]
                               - r["travel_series"][0]) for r in runs],
        per_seed_travel_episode=[r["travel_y"] for r in runs],
        per_seed_fell=[int(r["fell"]) for r in runs],
    )


def add_env_flags(ap: argparse.ArgumentParser) -> None:
    # same plant flags as the trainer (they go through env_kwargs)
    ap.add_argument("--assisted", action="store_true")
    ap.add_argument("--compliant", action="store_true",
                    help="score on the compliant-cuff model (models/human_exo.xml) "
                         "instead of the direct plant. Applied after --assisted")
    ap.add_argument("--exo-action", action="store_true")
    ap.add_argument("--gait-drive", action="store_true")
    ap.add_argument("--gait-params", default=None)
    ap.add_argument("--walk-progress-weight", type=float, default=0.0)
    ap.add_argument("--walk-target-speed", type=float, default=0.0)
    ap.add_argument("--offload", type=float, default=0.0)
    ap.add_argument("--joint-cap", type=float, default=None)
    ap.add_argument("--ankle-cap", type=float, default=None)
    ap.add_argument("--exo-residual-scale", type=float, nargs=6, default=None)
    ap.add_argument("--free-ankle", action="store_true")
    ap.add_argument("--stabilize-foot", action="store_true")
    ap.add_argument("--decimation", type=int, default=20)
    ap.add_argument("--exo-decimation", type=int, default=1)
    ap.add_argument("--episode-seconds", type=float, default=12.0)
    ap.add_argument("--settle-decisions", type=int, default=120)
    ap.add_argument("--human-effort-weight", type=float, default=0.05)
    ap.add_argument("--survival-bonus", type=float, default=1.0)
    ap.add_argument("--allocator-solver", default="bvls")
    ap.add_argument("--hip-adduction-cap", type=float, default=None)
    # Needed by env_kwargs; unused when walking.
    ap.add_argument("--step-bonus", type=float, default=0.0)
    ap.add_argument("--step-window", type=float, nargs=2, default=(-0.15, -0.05))
    ap.add_argument("--recovery-bonus", type=float, default=0.0)
    ap.add_argument("--step-recovery-bonus", type=float, default=0.0)
    ap.add_argument("--recovery-speed-bonus", type=float, default=0.0)
    ap.add_argument("--push-bodies", nargs="*", default=None)
    ap.add_argument("--push-axes", type=int, nargs="*", default=None)
    ap.add_argument("--push-omnidirectional", action="store_true")


def walking_env_kwargs(args) -> dict:
    # env_kwargs plus the walking setup. view_policy uses this too, so the
    # viewer shows the same plant that got scored.
    kw = env_kwargs(args)
    if args.compliant:
        # Swap in the compliant-cuff plant (a transfer test; nothing trains on it).
        kw["model_path"] = "models/human_exo.xml"
        kw["calibration_path"] = "models/exo_transmission_calibration.json"
        kw["exo_rated_torque_nm"] = None
    # No pushes when walking.
    kw.update(push_force_range=(0.0, 0.0), push_bodies=("torso",), push_axes=(0,))
    return kw


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_env_flags(ap)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--save-arm", default=None,
                    help="save raw per-decision rollouts for the given arms to this "
                         "file instead of printing a table; use --combine afterwards")
    ap.add_argument("--combine", nargs="+", default=None,
                    help="load saved arm files and print the table over their common "
                         "window")
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.combine:
        results, order = {}, []
        for path in args.combine:
            with open(path, "rb") as fh:
                d = pickle.load(fh)
            for name, runs in d["results"].items():
                results[name] = runs
                order.append(name)
        seeds = d["seeds"]
        report(results, order, seeds, args.decimation, args.out)
        return 0

    if not args.arms:
        ap.error("--arms is required unless --combine is given")
    base = walking_env_kwargs(args)

    seeds = [args.seed0 + i for i in range(args.seeds)]
    arms = [parse_arm(a, args.offload) for a in args.arms]
    max_decisions = int(args.episode_seconds
                        / (args.decimation * 0.001)) + 10

    results = {}
    for arm in arms:
        kw = dict(base)
        kw["human_offload"] = arm["offload"]
        env = SteppingEnv(**kw)
        model = None
        if arm["model"]:
            from stable_baselines3 import PPO
            model = PPO.load(arm["model"], device="cpu")
        results[arm["name"]] = [rollout(env, model, s, max_decisions)
                                for s in seeds]
        print("ran %-34s %d seeds" % (arm["name"], len(seeds)), flush=True)
        # Free each arm's env and policy before the next one to save RAM.
        del env, model
        gc.collect()

    if args.save_arm:
        with open(args.save_arm, "wb") as fh:
            pickle.dump({"results": results, "seeds": seeds}, fh)
        print("saved %d arm(s) to %s" % (len(results), args.save_arm))
        return 0
    report(results, [a["name"] for a in arms], seeds, args.decimation,
           args.out)
    return 0


def report(results: dict, order: list, seeds: list, decimation: int,
           out: str | None) -> None:
    window = min(r["decisions"] for runs in results.values() for r in runs)
    print("\ncommon window: %d decisions (%.2f s) across %d arms x %d seeds"
          % (window, window * decimation * 0.001, len(order), len(seeds)))
    print("\n%-34s | human_leg | ankle |   exo | share | root_hold | sat  | "
          "cuff_pk | trav_win | trav_ep | secs |  td  | fell" % "arm")
    rows = {}
    ref = None
    for name in order:
        s = score(results[name], window)
        if ref is None:
            ref = s["human_leg_nm"]
        s["human_change_pct"] = 100.0 * (s["human_leg_nm"] - ref) / max(ref, 1e-9)
        rows[name] = s
        print("%-34s | %9.3f | %5.2f | %5.2f | %5.3f | %9.1f | %4.2f | %7.1f | "
              "%8.3f | %7.3f | %4.1f | %4.1f | %d/%d  (%+.1f%% human)"
              % (name, s["human_leg_nm"], s["human_ankle_nm"],
                 s["exo_nm"], s["exo_share"], s["root_hold_nm"],
                 s["root_sat_frac"], s["cuff_peak_n"], s["travel_window_m"],
                 s["travel_y"], s["seconds"], s["touchdowns"], s["fell"],
                 len(seeds), s["human_change_pct"]))

    # Paired against the first arm (the control), counting seeds that moved.
    if len(order) > 1:
        ctrl = order[0]
        print("\npaired against %s (same seeds, same plant, same gait):" % ctrl)
        c = rows[ctrl]
        for name in order[1:]:
            s = rows[name]
            dh = np.array(s["per_seed_human"]) - np.array(c["per_seed_human"])
            dt = np.array(s["per_seed_travel"]) - np.array(c["per_seed_travel"])
            df = np.array(s["per_seed_fell"]) - np.array(c["per_seed_fell"])
            print("  %-32s human %+7.3f N*m on %2d/%d seeds lower | "
                  "travel %+6.3f m on %2d/%d seeds further | falls %+d"
                  % (name, float(dh.mean()), int((dh < 0).sum()),
                     len(seeds), float(dt.mean()), int((dt > 0).sum()),
                     len(seeds), int(df.sum())))

    if out:
        Path(out).write_text(json.dumps(
            {"window": window, "seeds": seeds, "rows": rows}, indent=2))
        print("\nwrote " + out)


if __name__ == "__main__":
    raise SystemExit(main())
