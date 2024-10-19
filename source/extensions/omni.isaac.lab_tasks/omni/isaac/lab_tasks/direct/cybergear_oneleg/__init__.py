# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents, cybergear_oneleg_env

##
# Register Gym environments.
##

gym.register(
    id="cybergear_oneleg_jumping-v0",
    entry_point="omni.isaac.lab_tasks.direct.cybergear_oneleg.cybergear_oneleg_env:CybergearOneLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cybergear_oneleg_env.CybergearOneLegEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:CybergearOneLegPPORunnerCfg",
        # "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        # "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
