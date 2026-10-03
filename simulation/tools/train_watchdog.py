"""Restarts train_stepping_ppo.py from the latest checkpoint if a worker gets
killed.

    python tools/train_watchdog.py --checkpoint-dir DIR --target-steps N
        --log-prefix logs/myrun -- <args for scripts/train_stepping_ppo.py>

runs\\run_walking_decim5_20260916.cmd is a full example.

Everything after -- is passed to the trainer, except --resume and --timesteps
which the watchdog fills in (newest checkpoint, steps left).

Only retries when it looks like a resource kill (broken pipe, WinError
109/1450/1455, OOM) AND the last attempt got further. Normal errors just stop.

It always resumes from the newest checkpoint in --checkpoint-dir. To restart
from an earlier one, move the later checkpoints out of the directory.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path

STEP_RE = re.compile(r"_(\d+)_steps\.zip$")


@contextmanager
def prevent_idle_sleep():
    # stop windows from going to sleep mid-run
    if sys.platform != "win32":
        yield
        return
    import ctypes
    set_state = ctypes.windll.kernel32.SetThreadExecutionState
    set_state.argtypes = [ctypes.c_uint]
    set_state.restype = ctypes.c_uint
    if not set_state(0x80000001):  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        raise OSError("could not suppress Windows idle sleep")
    try:
        yield
    finally:
        set_state(0x80000000)     # release this thread's request


def saved_steps(path: Path) -> int:
    # read the step count straight from the zip, no torch needed
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f"corrupt policy archive: {path}")
        data = json.loads(archive.read("data"))
    steps = data["num_timesteps"]
    if type(steps) is not int or steps < 0:
        raise ValueError(f"invalid num_timesteps in {path}")
    return steps


def trainer_option(arguments: list[str], name: str) -> str | None:
    value = None
    for index, argument in enumerate(arguments):
        if argument == name:
            if index + 1 >= len(arguments):
                raise ValueError(f"{name} requires a value")
            value = arguments[index + 1]
        elif argument.startswith(name + "="):
            value = argument.split("=", 1)[1]
    return value


def policy_path(value: str) -> Path:
    path = Path(value)
    return path if path.suffix == ".zip" else Path(str(path) + ".zip")


def newest_checkpoint(directory: Path) -> tuple[Path | None, int]:
    # Go by the step count in the filename, not mtime.
    candidates = []
    for p in directory.glob("*_steps.zip"):
        m = STEP_RE.search(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    for steps, p in sorted(candidates, reverse=True):
        try:
            actual = saved_steps(p)
            if actual != steps:
                raise ValueError(f"filename says {steps}, archive says {actual}")
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError) as exc:
            print(f"[watchdog] skipping unreadable checkpoint {p}: {exc}", flush=True)
            continue
        return p, steps
    return None, 0


def looks_like_resource_kill(returncode: int, err_path: Path) -> bool:
    if returncode == 0:
        return False
    try:
        tail = err_path.read_text(errors="replace")[-8000:]
    except OSError:
        return True          # no stderr at all reads as an abrupt death
    if "MuJoCo contact/constraint allocation exhausted" in tail:
        # Arena budget too small; a restart would hit it again.
        return False
    signatures = ("EOFError", "BrokenPipeError", "WinError 109",
                  "Could not allocate memory", "MemoryError",
                  "FatalError", "ConnectionResetError",
                  # 1450/1455: Windows commit exhaustion. Text is matched too
                  # since the message is localised.
                  "WinError 1450", "WinError 1455",
                  "Insufficient system resources")
    return any(s in tail for s in signatures)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--target-steps", type=int, required=True,
                    help="absolute total_timesteps to stop at")
    ap.add_argument("--log-prefix", required=True,
                    help="attempt N writes <prefix>_aN.log / .err")
    ap.add_argument("--max-restarts", type=int, default=12)
    ap.add_argument("--reset-log-std-first", type=float, default=None,
                    help="pass --reset-log-std VALUE on the first attempt only")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- then the trainer's own arguments")
    args = ap.parse_args()

    passthrough = args.rest[1:] if args.rest[:1] == ["--"] else args.rest
    if not passthrough:
        ap.error("nothing to run: put the trainer's arguments after --")
    if args.target_steps <= 0 or args.max_restarts < 0:
        ap.error("target-steps must be positive and max-restarts nonnegative")
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    output = policy_path(trainer_option(passthrough, "--output")
                         or "models/stepping_ppo")
    # Don't retrain if the final policy already reached the target.
    if output.is_file():
        try:
            completed = saved_steps(output)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile, EOFError):
            completed = -1
        if completed >= args.target_steps:
            print(f"[watchdog] final policy already reached {completed}: {output}",
                  flush=True)
            return 0

    for attempt in range(args.max_restarts + 1):
        ckpt, steps = newest_checkpoint(ckpt_dir)
        if ckpt is None:
            initial = trainer_option(passthrough, "--resume")
            if initial is not None:
                ckpt = policy_path(initial)
                steps = saved_steps(ckpt)
        remaining = max(0, args.target_steps - steps)

        cmd = [args.python, "scripts/train_stepping_ppo.py", *passthrough,
               "--checkpoint-dir", str(ckpt_dir), "--timesteps", str(remaining)]
        if ckpt is not None:
            cmd += ["--resume", str(ckpt)]
        if attempt == 0 and args.reset_log_std_first is not None:
            cmd += ["--reset-log-std", str(args.reset_log_std_first)]
        log_index = attempt
        while (Path(f"{args.log_prefix}_a{log_index}.log").exists()
               or Path(f"{args.log_prefix}_a{log_index}.err").exists()):
            log_index += 1
        log = Path(f"{args.log_prefix}_a{log_index}.log")
        err = Path(f"{args.log_prefix}_a{log_index}.err")
        log.parent.mkdir(parents=True, exist_ok=True)
        print(f"[watchdog] attempt {attempt}: from {steps} steps, "
              f"{remaining} remaining -> {log}", flush=True)

        with open(log, "w") as lo, open(err, "w") as le, prevent_idle_sleep():
            rc = subprocess.run(cmd, stdout=lo, stderr=le).returncode

        _c, now = newest_checkpoint(ckpt_dir)
        print(f"[watchdog] attempt {attempt} exited {rc} at ~{now} steps",
              flush=True)
        if rc == 0:
            print("[watchdog] clean exit, run complete", flush=True)
            return 0
        if not looks_like_resource_kill(rc, err):
            print(f"[watchdog] exit {rc} does not look like a resource kill; "
                  f"not retrying. See {err}", flush=True)
            return rc
        if now <= steps:
            print("[watchdog] no progress since the last attempt; stopping "
                  "rather than looping on a config that cannot start",
                  flush=True)
            return rc
        print("[watchdog] resource kill with progress made; resuming in 20s",
              flush=True)
        time.sleep(20)

    print("[watchdog] out of restarts", flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
