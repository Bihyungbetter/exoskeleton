"""SubprocVecEnv but the workers don't import torch.

We're memory bound on worker count and every SB3 import drags in torch
(~200MB each) even though the workers never use it. Parent side is just SB3's
SubprocVecEnv, only the spawning is different. Worker code is in lean_worker,
and it gets the env factory as raw cloudpickle bytes because SB3's
CloudpickleWrapper would import torch too.

Gotchas:
- env factory has to return a plain env, not Monitor-wrapped (trainer uses
  VecMonitor)
- on Windows the child re-imports the launching script, so that script can't
  import stable_baselines3 at the top (train_stepping_ppo does it in main())
"""
from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Callable

import cloudpickle
import gymnasium as gym
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

from envs.lean_worker import worker


class LeanSubprocVecEnv(SubprocVecEnv):
    def __init__(self, env_fns: list[Callable[[], gym.Env]],
                 start_method: str | None = None):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)
        if not n_envs:
            raise ValueError("at least one environment is required")
        if start_method is None:
            forkserver_available = "forkserver" in mp.get_all_start_methods()
            start_method = "forkserver" if forkserver_available else "spawn"
        ctx = mp.get_context(start_method)

        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(n_envs)], strict=True)
        self.processes = []
        try:
            for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes,
                                                   env_fns, strict=True):
                args = (work_remote, remote, cloudpickle.dumps(env_fn))
                process = ctx.Process(target=worker, args=args, daemon=True)
                process.start()
                self.processes.append(process)
                work_remote.close()
                if len(self.processes) == 1:
                    # Let the first worker fill the MJB cache before the rest
                    # start, so they don't all pay the XML compiler's memory.
                    remote.send(("get_spaces", None))
                    observation_space, action_space = remote.recv()

            # Skip SubprocVecEnv.__init__, which would spawn SB3's workers.
            VecEnv.__init__(self, n_envs, observation_space, action_space)
        except BaseException:
            self._terminate_workers()
            raise

    def _terminate_workers(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
        for remote in (*self.remotes, *self.work_remotes):
            remote.close()
        self.closed = True

    def close(self) -> None:
        # don't hang forever if a worker died
        if self.closed:
            return
        deadline = time.monotonic() + 5.0
        try:
            if self.waiting:
                for remote in self.remotes:
                    if not remote.poll(max(0.0, deadline - time.monotonic())):
                        return
                    remote.recv()
                self.waiting = False
            for remote in self.remotes:
                remote.send(("close", None))
            for process in self.processes:
                process.join(timeout=max(0.0, deadline - time.monotonic()))
        except (EOFError, OSError):
            pass
        finally:
            self._terminate_workers()

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper],
                       indices: VecEnvIndices = None) -> list[bool]:
        # compare by name so the child doesn't have to unpickle an SB3 class
        target_remotes = self._get_target_remotes(indices)
        name = f"{wrapper_class.__module__}.{wrapper_class.__qualname__}"
        for remote in target_remotes:
            remote.send(("is_wrapped_name", name))
        return [remote.recv() for remote in target_remotes]
