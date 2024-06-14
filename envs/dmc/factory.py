from envs.dmc.suite.suite_env import SuiteBaseEnv
from envs.dmc.hurdle_envs.hurdle_walker import HurdleWalker
from envs.dmc.hurdle_envs.hurdle_cheetah import HurdleCheetah
from envs.dmc.corridor_ant.corridor_ant import AntEmptyCorridorEnv, AntWallsCorridorEnv
from envs.dmc.dmc_mbrl_env import AbstractBaseEnv


def build_dmc_base_env(domain_name: str,
                       task_name: str,
                       seed: int,
                       env_kwargs: dict = {}) -> AbstractBaseEnv:
    if domain_name.startswith("hurdle"):
        if domain_name == "hurdle_cheetah":
            return HurdleCheetah(task=task_name,
                                 seed=seed,
                                 **env_kwargs)
        elif domain_name == "hurdle_walker":
            return HurdleWalker(task=task_name,
                                seed=seed,
                                **env_kwargs)
        else:
            raise ValueError(f"Unknown domain {domain_name}")
    elif domain_name == "ant":
        if task_name == "empty":
            return AntEmptyCorridorEnv(seed=seed,
                                       **env_kwargs)
        elif task_name == "walls":
            return AntWallsCorridorEnv(seed=seed,
                                       **env_kwargs)
        else:
            raise ValueError(f"Unknown domain {domain_name}")
    else:
        return SuiteBaseEnv(domain_name=domain_name,
                            task_name=task_name,
                            seed=seed)