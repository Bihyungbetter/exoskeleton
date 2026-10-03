"""Worker for LeanSubprocVecEnv.

DON'T import torch or stable_baselines3 here, not even indirectly - any SB3
import runs its __init__ which loads torch.

Same protocol as SB3's worker plus `is_wrapped_name` (takes a class name
instead of a pickled class, since unpickling Monitor would pull in torch).
"""
from __future__ import annotations

import multiprocessing as mp

import cloudpickle
import gymnasium as gym


def _is_wrapped_by_name(env: gym.Env, qualified_name: str) -> bool:
    while isinstance(env, gym.Wrapper):
        cls = type(env)
        if f"{cls.__module__}.{cls.__qualname__}" == qualified_name:
            return True
        env = env.env
    return False


def worker(remote: mp.connection.Connection,
           parent_remote: mp.connection.Connection,
           env_fn_bytes: bytes) -> None:
    parent_remote.close()
    env = cloudpickle.loads(env_fn_bytes)()
    reset_info: dict | None = {}
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info["TimeLimit.truncated"] = truncated and not terminated
                if done:
                    info["terminal_observation"] = observation
                    observation, reset_info = env.reset()
                remote.send((observation, reward, done, info, reset_info))
            elif cmd == "reset":
                maybe_options = {"options": data[1]} if data[1] else {}
                observation, reset_info = env.reset(seed=data[0], **maybe_options)
                remote.send((observation, reset_info))
            elif cmd == "render":
                remote.send(env.render())
            elif cmd == "close":
                env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((env.observation_space, env.action_space))
            elif cmd == "env_method":
                method = env.get_wrapper_attr(data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == "get_attr":
                remote.send(env.get_wrapper_attr(data))
            elif cmd == "has_attr":
                try:
                    env.get_wrapper_attr(data)
                    remote.send(True)
                except AttributeError:
                    remote.send(False)
            elif cmd == "set_attr":
                remote.send(setattr(env, data[0], data[1]))
            elif cmd == "is_wrapped_name":
                remote.send(_is_wrapped_by_name(env, data))
            else:
                raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break
