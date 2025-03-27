import numpy as np
import pybullet as p

from .env import AssistiveEnv
from .agents import furniture
from .agents.furniture import Furniture

class BedBathingEnv(AssistiveEnv):
    def __init__(self, robot, human):
        super(BedBathingEnv, self).__init__(robot=robot, human=human, task='bed_bathing', obs_robot_len=(17 + len(robot.controllable_joint_indices) - (len(robot.wheel_joint_indices) if robot.mobile else 0)), obs_human_len=(18 + len(human.controllable_joint_indices)))
    
    #新代码 pref
    def human_preferences(self, estimated_weights, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
        
        # Slow end effector velocities
        reward_velocity = -end_effector_velocity

        # < 10 N force at target
        reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

        reward_force_nontarget = -(total_force_on_human - tool_force_at_target)
        # Penalty when robot spills food on the person
        reward_food_hit_human = food_hit_human_reward
        # Human prefers food entering mouth at low velocities
        reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)


        return reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities

    # def human_preferences(self, end_effector_velocity=0, total_force_on_human=0, tool_force_at_target=0, food_hit_human_reward=0, food_mouth_velocities=[], dressing_forces=[[]], arm_manipulation_tool_forces_on_human=[0, 0], arm_manipulation_total_force_on_human=0):
    #     # Slow end effector velocities
    #     reward_velocity = -end_effector_velocity

    #     # < 10 N force at target
    #     reward_high_target_forces = 0 if tool_force_at_target < 10 else -tool_force_at_target

    #     # --- Scratching, Wiping ---
    #     # Any force away from target is low
    #     reward_force_nontarget = -(total_force_on_human - tool_force_at_target)

    #     # Penalty when robot spills food on the person
    #     reward_food_hit_human = food_hit_human_reward
    #     # Human prefers food entering mouth at low velocities
    #     reward_food_velocities = 0 if len(food_mouth_velocities) == 0 else -np.sum(food_mouth_velocities)


    #     return self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities


    #改动
    def step(self, action, args, estimated_weights, success_rate):
        if self.human.controllable:
            action = np.concatenate([action['robot'], action['human']])
        self.take_step(action)

        obs = self._get_obs()

        # Get human preferences
        end_effector_velocity = np.linalg.norm(self.robot.get_velocity(self.robot.left_end_effector))
        
        #新代码，这里只有项，出来之后再处理出各自的reward
        reward_velocity, reward_force_nontarget, reward_high_target_forces, reward_food_hit_human, reward_food_velocities = \
            self.human_preferences(estimated_weights=estimated_weights, # 传递这个参数
                                   end_effector_velocity=end_effector_velocity, 
                                   total_force_on_human=self.total_force_on_human, 
                                   tool_force_at_target=self.tool_force_on_human)
        
        # prefv, preff, prefh, prefhit, preffv = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_on_human)
        
        # 这个矩阵是为了给robot学习，所以要进行归一化
        normed_gt = estimated_weights.get_normalized_gt_weights()
        # print(normed_gt)
        v,f,h,hit,fv = normed_gt['C_v']*reward_velocity, normed_gt['C_f']*reward_force_nontarget, normed_gt['C_hf']*reward_high_target_forces, normed_gt['C_fd']*reward_food_hit_human, normed_gt['C_fdv']*reward_food_velocities
        pref_array = [v,f,h,hit,fv] 
        pref_array = np.array(pref_array).reshape(1, 5)

        #这里要用放缩后的值去计算robot的pref reward
        denormed_estimate_w = estimated_weights.get_denormed_weights()
        prefv_r, preff_r, prefh_r, prefhit_r, preffv_r = denormed_estimate_w['C_v']*reward_velocity, denormed_estimate_w['C_f']*reward_force_nontarget, denormed_estimate_w['C_hf']*reward_high_target_forces, denormed_estimate_w['C_fd']*reward_food_hit_human, denormed_estimate_w['C_fdv']*reward_food_velocities
        prefv, preff, prefh, prefhit, preffv = self.C_v*reward_velocity, self.C_f*reward_force_nontarget, self.C_hf*reward_high_target_forces, self.C_fd*reward_food_hit_human, self.C_fdv*reward_food_velocities
        preferences_score = prefv + preff + prefh + prefhit + preffv
        
        #social 专用 preferences_score_r
        preferences_score_r = prefv_r + preff_r + prefh_r + prefhit_r + preffv_r
        #新代码，这样做是为了确保机器人在学习到成功的策略基础上，才综合学习preference相关内容
        preferences_score_r = np.random.choice([preferences_score_r, 0], p=[success_rate, 1 - success_rate])

        
        # preferences_score = self.human_preferences(end_effector_velocity=end_effector_velocity, total_force_on_human=self.total_force_on_human, tool_force_at_target=self.tool_force_on_human)

        reward_distance = -min(self.tool.get_closest_points(self.human, distance=5.0)[-1])
        reward_action = -np.linalg.norm(action) # Penalize actions
        reward_new_contact_points = self.new_contact_points # Reward new contact points on a person

        human_dist_reward =  self.config('distance_weight')*reward_distance
        human_action_reward = self.config('action_weight')*reward_action
        human_food_reward = self.config('wiping_reward_weight')*reward_new_contact_points
        human_pref_reward = preferences_score
        robot_dist_reward =  self.config('distance_weight')*reward_distance
        robot_action_reward = self.config('action_weight')*reward_action
        robot_food_reward =  self.config('wiping_reward_weight')*reward_new_contact_points
        # robot_pref_reward =  preferences_score
        if args.algo == 'PPO':
            if args.social:
                if args.dempref:
                    robot_pref_reward = preferences_score_r # baseline改这里
                else:
                    robot_pref_reward = 0
            else:
                if args.given_pref:
                    robot_pref_reward = preferences_score
                else:
                    robot_pref_reward = 0
        else:
            robot_pref_reward = 0
        
        # robot_pref_reward = prefv + preff + prefh
        # to args, split vars: reward_distance_mouth_target, preferences_score
        reward_human = human_dist_reward + human_action_reward + human_food_reward + human_pref_reward
        reward_robot = robot_dist_reward + robot_action_reward + robot_food_reward + robot_pref_reward
        reward = 0.5 * (reward_human + reward_robot)

        # reward = self.config('distance_weight')*reward_distance + self.config('action_weight')*reward_action + self.config('wiping_reward_weight')*reward_new_contact_points + preferences_score

        if self.gui and self.tool_force_on_human > 0:
            print('Task success:', self.task_success, 'Force at tool on human:', self.tool_force_on_human, reward_new_contact_points)

        info = {'particles' : self.task_success, 'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= self.total_target_count*self.config('task_success_threshold')), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        
        # info = {'total_force_on_human': self.total_force_on_human, 'task_success': int(self.task_success >= (self.total_target_count*self.config('task_success_threshold'))), 'action_robot_len': self.action_robot_len, 'action_human_len': self.action_human_len, 'obs_robot_len': self.obs_robot_len, 'obs_human_len': self.obs_human_len}
        done = self.iteration >= 200

        if not self.human.controllable:
            return obs, reward, done, info
        else:
            # Co-optimization with both human and robot controllable
            # return obs, {'robot': reward_robot, 'human': reward_human, '__all__':reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}, {'human_dist_reward': human_dist_reward, 'human_action_reward': human_action_reward, 'human_food_reward': human_food_reward, 'human_pref_reward' : human_pref_reward,
            # 'robot_dist_reward': robot_dist_reward, 'robot_action_reward': robot_action_reward, 'robot_food_reward': robot_food_reward, 'robot_pref_reward' : robot_pref_reward, 'vel': prefv, 'force': preff, 'h_force' : prefh, 'hit' : prefhit, 'food_v' : preffv}
            return obs, {'robot': reward_robot, 'human': reward_human, '__all__':reward}, {'robot': done, 'human': done, '__all__': done}, {'robot': info, 'human': info}, {'human_dist_reward': human_dist_reward, 'human_action_reward': human_action_reward, 'human_food_reward': human_food_reward, 'human_pref_reward' : human_pref_reward,
            'robot_dist_reward': robot_dist_reward, 'robot_action_reward': robot_action_reward, 'robot_food_reward': robot_food_reward, 'robot_pref_reward' : robot_pref_reward, 'vel': prefv, 'force': preff, 'h_force' : prefh, 'hit' : prefhit, 'food_v' : preffv}, pref_array

    def get_total_force(self):
        total_force_on_human = np.sum(self.robot.get_contact_points(self.human)[-1])
        tool_force = np.sum(self.tool.get_contact_points()[-1])
        tool_force_on_human = 0
        new_contact_points = 0
        for linkA, linkB, posA, posB, force in zip(*self.tool.get_contact_points(self.human)):
            total_force_on_human += force
            if linkA in [1]:
                tool_force_on_human += force
                # Only consider contact with human upperarm, forearm, hand
                if linkB < 0 or linkB > len(self.human.all_joint_indices):
                    continue

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_upperarm_world, self.targets_upperarm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_upperarm = [t for i, t in enumerate(self.targets_pos_on_upperarm) if i not in indices_to_delete]
                self.targets_upperarm = [t for i, t in enumerate(self.targets_upperarm) if i not in indices_to_delete]
                self.targets_pos_upperarm_world = [t for i, t in enumerate(self.targets_pos_upperarm_world) if i not in indices_to_delete]

                indices_to_delete = []
                for i, (target_pos_world, target) in enumerate(zip(self.targets_pos_forearm_world, self.targets_forearm)):
                    if np.linalg.norm(posB - target_pos_world) < 0.025:
                        # The robot made contact with a point on the person's arm
                        new_contact_points += 1
                        self.task_success += 1
                        target.set_base_pos_orient(self.np_random.uniform(1000, 2000, size=3), [0, 0, 0, 1])
                        indices_to_delete.append(i)
                self.targets_pos_on_forearm = [t for i, t in enumerate(self.targets_pos_on_forearm) if i not in indices_to_delete]
                self.targets_forearm = [t for i, t in enumerate(self.targets_forearm) if i not in indices_to_delete]
                self.targets_pos_forearm_world = [t for i, t in enumerate(self.targets_pos_forearm_world) if i not in indices_to_delete]

        return tool_force, tool_force_on_human, total_force_on_human, new_contact_points

    def _get_obs(self, agent=None):
        tool_pos, tool_orient = self.tool.get_pos_orient(1)
        tool_pos_real, tool_orient_real = self.robot.convert_to_realworld(tool_pos, tool_orient)
        robot_joint_angles = self.robot.get_joint_angles(self.robot.controllable_joint_indices)
        # Fix joint angles to be in [-pi, pi]
        robot_joint_angles = (np.array(robot_joint_angles) + np.pi) % (2*np.pi) - np.pi
        if self.robot.mobile:
            # Don't include joint angles for the wheels
            robot_joint_angles = robot_joint_angles[len(self.robot.wheel_joint_indices):]
        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]
        shoulder_pos_real, _ = self.robot.convert_to_realworld(shoulder_pos)
        elbow_pos_real, _ = self.robot.convert_to_realworld(elbow_pos)
        wrist_pos_real, _ = self.robot.convert_to_realworld(wrist_pos)
        self.tool_force, self.tool_force_on_human, self.total_force_on_human, self.new_contact_points = self.get_total_force()
        robot_obs = np.concatenate([tool_pos_real, tool_orient_real, robot_joint_angles, shoulder_pos_real, elbow_pos_real, wrist_pos_real, [self.tool_force]]).ravel()
        if agent == 'robot':
            return robot_obs
        if self.human.controllable:
            human_joint_angles = self.human.get_joint_angles(self.human.controllable_joint_indices)
            tool_pos_human, tool_orient_human = self.human.convert_to_realworld(tool_pos, tool_orient)
            shoulder_pos_human, _ = self.human.convert_to_realworld(shoulder_pos)
            elbow_pos_human, _ = self.human.convert_to_realworld(elbow_pos)
            wrist_pos_human, _ = self.human.convert_to_realworld(wrist_pos)
            human_obs = np.concatenate([tool_pos_human, tool_orient_human, human_joint_angles, shoulder_pos_human, elbow_pos_human, wrist_pos_human, [self.total_force_on_human, self.tool_force_on_human]]).ravel()
            if agent == 'human':
                return human_obs
            # Co-optimization with both human and robot controllable
            return {'robot': robot_obs, 'human': human_obs}
        return robot_obs

    def reset(self):
        super(BedBathingEnv, self).reset()
        self.build_assistive_env('bed', fixed_human_base=False)

        self.furniture.set_friction(self.furniture.base, friction=5)

        # Setup human in the air and let them settle into a resting pose on the bed
        joints_positions = [(self.human.j_right_shoulder_x, 30)]
        self.human.setup_joints(joints_positions, use_static_joints=False, reactive_force=None)
        self.human.set_base_pos_orient([-0.15, 0.2, 0.95], [-np.pi/2.0, 0, 0])

        p.setGravity(0, 0, -1, physicsClientId=self.id)

        # Add small variation in human joint positions
        motor_indices, motor_positions, motor_velocities, motor_torques = self.human.get_motor_joint_states()
        self.human.set_joint_angles(motor_indices, self.np_random.uniform(-0.1, 0.1, size=len(motor_indices)))

        # Let the person settle on the bed
        for _ in range(100):
            p.stepSimulation(physicsClientId=self.id)

        # Lock human joints and set velocities to 0
        joints_positions = []
        self.human.setup_joints(joints_positions, use_static_joints=True, reactive_force=None, reactive_gain=0.01)
        self.human.set_mass(self.human.base, mass=0)
        self.human.set_base_velocity(linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

        shoulder_pos = self.human.get_pos_orient(self.human.right_shoulder)[0]
        elbow_pos = self.human.get_pos_orient(self.human.right_elbow)[0]
        wrist_pos = self.human.get_pos_orient(self.human.right_wrist)[0]

        # Initialize the tool in the robot's gripper
        self.tool.init(self.robot, self.task, self.directory, self.id, self.np_random, right=False, mesh_scale=[1]*3)

        target_ee_pos = np.array([-0.6, 0.2, 1]) + self.np_random.uniform(-0.05, 0.05, size=3)
        target_ee_orient = self.get_quaternion(self.robot.toc_ee_orient_rpy[self.task])
        base_position = self.init_robot_pose(target_ee_pos, target_ee_orient, [(target_ee_pos, target_ee_orient)], [(shoulder_pos, None), (elbow_pos, None), (wrist_pos, None)], arm='left', tools=[self.tool], collision_objects=[self.human, self.furniture], wheelchair_enabled=False)

        if self.robot.wheelchair_mounted:
            # Load a nightstand in the environment for mounted arms
            self.nightstand = Furniture()
            self.nightstand.init('nightstand', self.directory, self.id, self.np_random)
            self.nightstand.set_base_pos_orient(np.array([-0.9, 0.7, 0]) + base_position, [0, 0, 0, 1])

        # Open gripper to hold the tool
        self.robot.set_gripper_open_position(self.robot.left_gripper_indices, self.robot.gripper_pos[self.task], set_instantly=True)

        self.generate_targets()

        p.setGravity(0, 0, -9.81, physicsClientId=self.id)
        if not self.robot.mobile:
            self.robot.set_gravity(0, 0, 0)
        self.human.set_gravity(0, 0, -1)
        self.tool.set_gravity(0, 0, 0)

        # Enable rendering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=self.id)

        self.init_env_variables()
        return self._get_obs()

    def generate_targets(self):
        self.target_indices_to_ignore = []
        if self.human.gender == 'male':
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.279, 0.043
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.257, 0.033
        else:
            self.upperarm, self.upperarm_length, self.upperarm_radius = self.human.right_shoulder, 0.264, 0.0355
            self.forearm, self.forearm_length, self.forearm_radius = self.human.right_elbow, 0.234, 0.027

        self.targets_pos_on_upperarm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.upperarm_length]), radius=self.upperarm_radius, distance_between_points=0.03)
        self.targets_pos_on_forearm = self.util.capsule_points(p1=np.array([0, 0, 0]), p2=np.array([0, 0, -self.forearm_length]), radius=self.forearm_radius, distance_between_points=0.03)

        self.targets_upperarm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_upperarm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.targets_forearm = self.create_spheres(radius=0.01, mass=0.0, batch_positions=[[0, 0, 0]]*len(self.targets_pos_on_forearm), visual=True, collision=False, rgba=[0, 1, 1, 1])
        self.total_target_count = len(self.targets_pos_on_upperarm) + len(self.targets_pos_on_forearm)
        self.update_targets()

    def update_targets(self):
        upperarm_pos, upperarm_orient = self.human.get_pos_orient(self.upperarm)
        self.targets_pos_upperarm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_upperarm, self.targets_upperarm):
            target_pos = np.array(p.multiplyTransforms(upperarm_pos, upperarm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_upperarm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

        forearm_pos, forearm_orient = self.human.get_pos_orient(self.forearm)
        self.targets_pos_forearm_world = []
        for target_pos_on_arm, target in zip(self.targets_pos_on_forearm, self.targets_forearm):
            target_pos = np.array(p.multiplyTransforms(forearm_pos, forearm_orient, target_pos_on_arm, [0, 0, 0, 1], physicsClientId=self.id)[0])
            self.targets_pos_forearm_world.append(target_pos)
            target.set_base_pos_orient(target_pos, [0, 0, 0, 1])

