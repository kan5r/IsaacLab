# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for cybergear one leg robot."""


import numpy as np
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets import ArticulationCfg

##
# Configuration
##

CYBERGEAR_ONELEG_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/workspace/IsaacLab/cybergear_one_leg_description/usd/cybergear_one_leg_with_slider_v2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2), 
        joint_pos={
            "joint_slider": 0.0, 
            "joint_hy": 0.0, 
            "joint_knee": 55.0 * np.pi / 180.0
        }
    ),
    actuators={
        "slider_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_slider"],
            effort_limit=400.0,
            velocity_limit=100.0,
            stiffness=0,
            damping=0.1,
            friction=10.0
        ),
        "hy_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_hy"],
            effort_limit=8.0,
            velocity_limit=100.0,
            stiffness=5.0,
            damping=0.5,
        ),
        "knee_actuator": ImplicitActuatorCfg(
            joint_names_expr=["joint_knee"], 
            effort_limit=8.0, 
            velocity_limit=100.0, 
            stiffness=5.0,
            damping=0.5
        ),
    },
)
"""Configuration for cybergear one leg robot."""
