# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.him_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class BipedalHimRoughCfg( LeggedRobotCfg ):
    class env( LeggedRobotCfg.env ):
        num_envs = 4096
        num_one_step_observations = 28  # command[:4], base_ang_vel, projected_gravity, pos(6-2), vel(6), actioins
        num_observations = num_one_step_observations * 6
        # num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187 # additional: base_lin_vel, external_forces, scan_dots
        num_one_step_privileged_obs = num_one_step_observations + 3 + 187 # additional: external_forces, scan_dots
        num_privileged_obs = num_one_step_privileged_obs * 1 # if not None a priviledge_obs_buf will be returned by step() (critic obs for assymetric training). None is returned otherwise 
        num_actions = 6
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 20 # episode length in seconds

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5]  # x,y,z [m]
        default_joint_angles = {  # target angles when action = 0.0
            # "L_thigh_joint": 1.22,
            # "L_calf_joint": -1.92,
            # "L_wheel_joint": 0.0,
            # "R_thigh_joint": 1.22,
            # "R_calf_joint": -1.92,
            # "R_wheel_joint": 0.0,
            "L_thigh_joint": 1.401,
            "L_calf_joint": -2.0717,
            "L_wheel_joint": 0.0,
            "R_thigh_joint": 1.401,
            "R_calf_joint": -2.0717,
            "R_wheel_joint": 0.0,
            # "L_thigh_joint": 0.9219,
            # "L_calf_joint": -1.4399,
            # "L_wheel_joint": 0.0,
            # "R_thigh_joint": 0.9219,
            # "R_calf_joint": -1.4399,
            # "R_wheel_joint": 0.0,
        }

    class control( LeggedRobotCfg.control ):
        control_type = 'MIXED_LIMBS_PV' # P: position, V: velocity, T: torques, MIXED_LIMBS_PV: position for some limbs, velocity for others
        pos_action_scale = 0.5
        vel_action_scale = 5.0
        # PD Drive parameters:
        position_control_indices = [0, 1, 3, 4] # indices of the joints that are controlled with position control
        velocity_control_indices = [2, 5]
        leg_kp = 25
        leg_kd = 0.3
        stiffness = {"L_thigh_joint": 25.0, "L_calf_joint": 25.0,"L_wheel_joint":0.0, 
                     "R_thigh_joint": 25.0, "R_calf_joint": 25.0,"R_wheel_joint":0.0}  # [N*m/rad]
        damping = {"L_thigh_joint": 0.3, "L_calf_joint": 0.3, "L_wheel_joint": 0.3, 
                   "R_thigh_joint": 0.3, "R_calf_joint": 0.3,"R_wheel_joint": 0.3}  # [N*m*s/rad]
        decimation = 4
        use_actuator_network = False
        hip_reduction = 1.0

    class commands( LeggedRobotCfg.commands ):
        curriculum = False
        max_curriculum = 3.0
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        init_height = 0.18
        class ranges:
            lin_vel_x = [-1.0, 1.0] # min max [m/s]
            lin_vel_y = [0.0, 0.0]   # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
            heading = [-3.14, 3.14]
            # height = [0.1, 0.25]

    class terrain( LeggedRobotCfg.terrain ):
        mesh_type = 'plane'
        vertical_scale = 0.002 # m

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/bipedal/urdf/quick_bipedal.urdf'
        name = "quick_bipedal"
        # foot_name = "foot"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False  # Some .obj meshes must be flipped from y-up to z-up
        # armature_joint = 0.01
        # armature_wheel = 0.006
        # damping_joint = 0.1
        # damping_wheel = 0.004
        # frictionloss_joint = 0.2
        # frictionloss_wheel = 0.05
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            tracking_lin_vel = 2.0
            # tracking_lin_vel_enhance = 1.0
            tracking_ang_vel = 1.0

            base_height = -20.0
            nominal_state_thigh = -0.2
            nominal_state_calf = -0.5
            joint_deviation = -0.2
            lin_vel_z = -1.0
            ang_vel_xy = -0.05
            orientation = -10.0

            dof_vel = -5e-5
            torque_limits = -0.1
            dof_acc = -2.5e-7
            torques = -1e-4
            action_rate = -0.01
            action_smooth = -0.01

            collision = -10.0
            dof_pos_limits = -10.0

        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        # clip_single_reward = 1
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = (
            0.97  # percentage of urdf limits, values above this limit are penalized
        )
        soft_dof_vel_limit = 1.0
        soft_torque_limit = 1.0
        base_height_target = 0.1871
        # base_height_target = 0.286
        # max_contact_force = 100.0  # forces above this value are penalized
        clearance_height_target = None

    class domain_rand( LeggedRobotCfg.domain_rand ):
        randomize_payload_mass = True
        payload_mass_range = [-1.5, 3]

        randomize_com_displacement = True
        com_displacement_range = [-0.05, 0.05]

        randomize_link_mass = True
        link_mass_range = [0.9, 1.1]
        
        randomize_friction = True
        friction_range = [0.2, 1.25]
        
        randomize_restitution = False
        restitution_range = [0., 1.0]
        
        randomize_motor_strength = True
        motor_strength_range = [0.9, 1.1]
        
        randomize_kp = True
        kp_range = [0.9, 1.1]
        
        randomize_kd = True
        kd_range = [0.9, 1.1]
        
        randomize_initial_joint_pos = True
        initial_joint_pos_range = [0.5, 1.5]
        
        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
        
        push_robots = True
        push_interval_s = 16
        max_push_vel_xy = 1.

        delay = True

        # randomize_armature = True
        # armature_range = [0.9, 1.1]

        # randomize_damping = True
        # armature_range = [0.9, 1.1]

        # randomize_frictionloss = True
        # frictionloss_range = [0.9, 1.1]

class BipedalHimRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'bipedal_him_plane'
        
        save_interval = 100 # check for potential saves every this many iterations
        max_iterations = 1500 # number of policy updates

  