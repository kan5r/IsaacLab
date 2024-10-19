# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from omni.isaac.lab_assets.cybergear_oneleg import CYBERGEAR_ONELEG_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import Articulation, ArticulationCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.math import sample_uniform
import omni.isaac.lab.envs.mdp as mdp


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 0.8),
            "dynamic_friction_range": (0.6, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (90.0, 110.0),
            "operation": "scale",
        },
    )

@configclass
class CybergearOneLegEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 1.0
    num_actions = 2
    num_observations = 9
    num_states = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = CYBERGEAR_ONELEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    slider_dof_name = "joint_slider"
    hip_dof_name = "joint_hy"
    knee_dof_name = "joint_knee"

    # slider_pos_offset = 0.45
    command_min = 0.0
    command_max = 1.0

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # reward scales
    rew_scale_g = 1.0
    rew_scale_h = -1.0
    rew_scale_j = -0.02
    rew_scale_jp = -0.02


class CybergearOneLegEnv(DirectRLEnv):
    cfg: CybergearOneLegEnvCfg

    def __init__(self, cfg: CybergearOneLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # joint position command
        self._actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)
        self._previous_actions = torch.zeros((self.num_envs, self.cfg.num_actions), device=self.device)

        # height position commands
        self._commands = torch.zeros(self.num_envs, 1, device=self.device)

        # specific joint indices
        self._slider_dof_idx, _ = self._robot.find_joints(self.cfg.slider_dof_name)
        self._hip_dof_idx, _ = self._robot.find_joints(self.cfg.hip_dof_name)
        self._knee_dof_idx, _ = self._robot.find_joints(self.cfg.knee_dof_name)
        self._actuation_dof_idx, _ = self._robot.find_joints((self.cfg.hip_dof_name, self.cfg.knee_dof_name))

        # specific body indices
        self._uleg_idx, _ = self._contact_sensor.find_bodies("uleg")
        # self._lleg_idx, _ = self._contact_sensor.find_bodies("lleg")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot_cfg)
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        # add articultion to scene
        self.scene.articulations["robot"] = self._robot
        # add contact sensor
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._actions = self.cfg.action_scale * actions.clone()

    def _apply_action(self):
        print(self._actions)
        # self._robot.set_joint_position_tar/get(self._actions, joint_ids=self._actuation_dof_idx)
        self._robot.set_joint_effort_target(, joint_ids=self._actuation_dof_idx)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        obs = torch.cat(
            (
                self._robot.data.joint_pos[:, self._hip_dof_idx[0]].unsqueeze(1),
                self._robot.data.joint_pos[:, self._knee_dof_idx[0]].unsqueeze(1),
                self._robot.data.joint_vel[:, self._hip_dof_idx[0]].unsqueeze(1),
                self._robot.data.joint_vel[:, self._knee_dof_idx[0]].unsqueeze(1),
                # self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1) \
                #     + self.cfg.slider_pos_offset,
                self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1),
                self._robot.data.joint_vel[:, self._slider_dof_idx[0]].unsqueeze(1),
                self._commands,
                self._previous_actions
            ),
            dim=-1
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        reward_energy = (
            torch.pow(self._robot.data.joint_vel[:, self._slider_dof_idx[0]], 2)
            + torch.pow(
                torch.where(
                    condition=self._robot.data.joint_pos[:, self._slider_dof_idx[0]] > 0.0,
                    input=self._robot.data.joint_pos_target[:, self._slider_dof_idx[0]],
                    other=0.0
                ),
                exponent=2
            )
        )

        # reward_height = torch.where(
        #     condition=self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1) + self.cfg.slider_pos_offset > self._commands,
        #     input=1-torch.exp(self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1) +self.cfg.slider_pos_offset - self._commands),
        #     other=0
        # ).squeeze(1)
        reward_height = torch.where(
            condition=self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1) > self._commands,
            input=1-torch.exp(self._robot.data.joint_pos[:, self._slider_dof_idx[0]].unsqueeze(1) - self._commands),
            other=0
        ).squeeze(1)


        reward_jerky = torch.sum(
            torch.pow(
                self._actions - self._previous_actions,
                exponent=2
            ),
            dim=1
        )

        # out_of_limits = -(
        #     self._robot.data.joint_pos[:, self._actuation_dof_idx] - self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 0]
        # ).clip(max=0.0)
        # out_of_limits += (
        #     self._robot.data.joint_pos[:, self._actuation_dof_idx] - self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 1]
        # ).clip(min=0.0)
        # reward_jp = out_of_limits

        reward_jp = torch.sum(
            torch.where(
                condition=torch.logical_and(
                    self._robot.data.joint_pos[:, self._actuation_dof_idx] >= self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 0], 
                    self._robot.data.joint_pos[:, self._actuation_dof_idx] <= self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 1]
                ),
                input=(
                    torch.exp(
                        -10 * (self._robot.data.joint_pos[:, self._actuation_dof_idx] - self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 0])
                    ) + torch.exp(
                        10 * ((self._robot.data.joint_pos[:, self._actuation_dof_idx] - self._robot.data.soft_joint_pos_limits[:, self._actuation_dof_idx, 1]))
                    )
                ),
                other=1.0
            ),
            dim=1
        )

        total_rewrd = (
            self.cfg.rew_scale_g * reward_energy 
            + self.cfg.rew_scale_h * reward_height 
            + self.cfg.rew_scale_j * reward_jerky 
            + self.cfg.rew_scale_jp * reward_jp
        )

        return total_rewrd

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._uleg_idx], dim=-1), dim=1)[0] > 0.1, # threshold: 0.1 N
            dim=1
        )
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]
                                                   ).uniform_(self.cfg.command_min, self.cfg.command_max)

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
