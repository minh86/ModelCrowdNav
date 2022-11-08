import logging
import itertools
import copy
import random
import numpy as np
from numpy.linalg import norm
import torch
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState, FullState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.utils import point_to_segment_dist


class DataGen(object):

    def __init__(self, memory, robot, env, policy):
        self.raw_memory = None
        self.action_space = None
        self.memory = memory
        self.robot = robot
        self.policy = robot.policy
        self.target_model = None
        self.env = env
        self.discomfort_penalty_factor = env.discomfort_penalty_factor
        self.discomfort_dist = env.discomfort_dist
        self.success_reward = env.success_reward
        self.collision_penalty = env.collision_penalty
        self.env.reset("train")
        px, py, vx, vy, radius, gx, gy, v_pref, theta = 0, 0, 0, 0, self.robot.radius, self.robot.gx, self.robot.gy, \
                                                        self.robot.v_pref, self.robot.theta
        self.full_state = FullState(px, py, vx, vy, radius, gx, gy, v_pref, theta)
        self.build_action_space()
        self.policy = policy

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def gen_new_episode(self, min_epi_length=30, max_epi_length=60, phase="train"):
        raw_states = []
        done = False
        i = 0
        ob = self.env.reset(phase)
        state = JointState(self.full_state, ob)
        raw_states.append(state)
        epi_length = random.randint(min_epi_length, max_epi_length)
        while not done and i < epi_length - 1:
            i += 1
            action = self.action_space[0]
            ob, reward, done, info = self.env.step(action)
            state = JointState(self.full_state, ob)
            raw_states.append(state)
        return raw_states

    def compute_position(self, self_state, action, delta_t):
        if self.robot.kinematics == 'holonomic':
            px = self_state.px + action.vx * delta_t
            py = self_state.py + action.vy * delta_t
        else:
            theta = self_state.theta + action.r
            px = self_state.px + np.cos(theta) * action.v * delta_t
            py = self_state.py + np.sin(theta) * action.v * delta_t

        return px, py

    def cal_reward(self, humans, self_state, action):
        dmin = float('inf')
        collision = False
        for i, human in enumerate(humans):
            px = human.px - self_state.px
            py = human.py - self_state.py
            # px = human.px
            # py = human.py
            if self.robot.kinematics == 'holonomic':
                vx = human.vx - action.vx
                vy = human.vy - action.vy
            else:
                vx = human.vx - action.v * np.cos(action.r + self_state.theta)
                vy = human.vy - action.v * np.sin(action.r + self_state.theta)
            ex = px + vx * self.policy.time_step
            ey = py + vy * self.policy.time_step
            # closest distance between boundaries of two agents
            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - human.radius - self.robot.radius
            if closest_dist < 0:
                collision = True
                break
            elif closest_dist < dmin:
                dmin = closest_dist
        # check if reaching the goal
        end_position = np.array(self.compute_position(self_state, action, self.policy.time_step))
        reaching_goal = norm(end_position - np.array(self.robot.get_goal_position())) < self.robot.radius

        if collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif reaching_goal:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif dmin < self.discomfort_dist:
            # only penalize agent for getting too close if it's visible
            # adjust the reward based on FPS
            reward = (dmin - self.discomfort_dist) * self.discomfort_penalty_factor * self.policy.time_step
            done = False
            info = Danger(dmin)
        else:
            reward = 0
            done = False
            info = Nothing()

        return reward, done, info

    def makeup_self_state(self, state, action, humans):  # create state with proper action
        vx, vy = 0, 0
        # get reverse action
        if isinstance(action, ActionXY):
            vx, vy = -action.vx, -action.vy
        if isinstance(action, ActionRot):
            next_theta = state.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)
            vx, vy = -next_vx, -next_vy
        state.vx, state.vy = vx, vy
        # get reward
        reward, done, info = self.cal_reward(humans, state, ActionXY(vx, vy))

        return state, reward, done, info

    def build_action_space(self):
        """
        Action space consists of 25 uniformly sampled actions in permitted range and 25 randomly sampled actions.
        """
        v_pref = self.full_state.v_pref
        holonomic = True if self.policy.kinematics == 'holonomic' else False
        speeds = [(np.exp((i + 1) / self.policy.speed_samples) - 1) / (np.e - 1) * v_pref for i in
                  range(self.policy.speed_samples)]
        if holonomic:
            rotations = np.linspace(0, np.pi, int(self.policy.rotation_samples / 2), endpoint=False)
        else:
            rotations = np.linspace(-np.pi / 4, np.pi / 4, self.policy.rotation_samples)

        action_space = [ActionXY(0, 0) if holonomic else ActionRot(0, 0)]
        for rotation, speed in itertools.product(rotations, speeds):
            if holonomic:
                action_space.append(ActionXY(speed * np.cos(rotation), speed * np.sin(rotation)))
            else:
                action_space.append(ActionRot(speed, rotation))

        self.action_space = action_space

    def pick_random_action(self):
        action = random.choice(self.action_space)
        return action

    def edit_episode(self, states, reach_goal=True):  # reach_goal or collision only
        reverse_states = states[::-1]
        states_with_action = []
        rewards = []
        # pick end position for robot
        if reach_goal:
            gx, gy = self.full_state.gx, self.full_state.gy
        else:
            hu = reverse_states[0].human_states[0]  # human start position is random, then pick the first one
            gx, gy = hu.px, hu.py
        self.full_state.px, self.full_state.py, self.full_state.vx, self.full_state.vy = gx, gy, 0, 0
        reverse_states = reverse_states[1:]
        for i, s in enumerate(reverse_states):
            action = self.pick_random_action()
            # get robot position from random action
            state_before = self.policy.propagate(self.full_state, action)
            reverse_states[i].self_state, reward, done, info = self.makeup_self_state(state_before, action,
                                                                                      reverse_states[i].human_states)
            self.full_state.px, self.full_state.py = \
                reverse_states[i].self_state.px, reverse_states[i].self_state.py
            # if done and i != 0:
            #     break
            states_with_action.append(self.transform(s))
            rewards.append(reward)

        return states_with_action[::-1], rewards[::-1]

    def correct_and_update(self, states, rewards, imitation_learning):
        if rewards[0] == self.success_reward or rewards[0] == self.collision_penalty:  # skip error episode
            return
        end_epi = 0
        error_case = True
        for i in range(len(rewards)):
            if rewards[i] == self.collision_penalty or rewards[i] == self.success_reward:
                end_epi = i + 1
                error_case = False
                break
        if error_case:
            return
        self.update_memory(states[:end_epi], rewards[:end_epi], imitation_learning)

    # gen data from imagination only
    def gen_new_data(self, num_sample, imitation_learning=False, reach_goal=True):
        for _ in range(num_sample):
            # Create random episode as robot stay the same position
            states = self.gen_new_episode()
            # start from end episode (goal or human (collision case)), make up random action for each state, collect reward
            states, rewards = self.edit_episode(states, reach_goal)
            # add data to memory
            self.correct_and_update(states, rewards, imitation_learning)

    def someone_is_moving(self, ob):
        min_speed = 1e-3
        for h in ob:
            if abs(h.vx) > min_speed or abs(h.vy) > min_speed:
                return True
        return False

    # get index for final state
    def get_episode_end_index(self):
        indexes = []
        for i, (ob, reward, done, info) in enumerate(self.raw_memory):
            if done:
                indexes.append(i)
        return indexes

    # get last real episode
    def pick_last_real_episode(self, random_epi=False):
        obs = []
        if random_epi:
            indexes = self.get_episode_end_index()
            i = random.choice(indexes)
            mem_states = self.raw_memory[:i]
        else:
            mem_states = self.raw_memory
        for i, (ob, reward, done, info) in enumerate(mem_states[::-1]):
            if self.someone_is_moving(ob):
                obs.append(ob)
            if done and i != 0:
                break
        return obs[::-1]

    # create JointState from last real episode
    def get_real_state(self, random_epi=False):
        obs = self.pick_last_real_episode(random_epi=random_epi)
        raw_states = []
        for ob in obs:
            state = JointState(self.full_state, ob)
            raw_states.append(state)
        return raw_states

    # add sim states to real states
    def create_imagine_on_real(self, states, min_end=7, min_sim_state=3, max_sim_state=15):
        length = random.randrange(min_end, len(states)-5)
        num_sim_state = random.randrange(min_sim_state, max_sim_state)
        raw_states = states[:length]
        human_states = raw_states[-1].human_states
        self.env.set_current_state(human_states)

        for _ in range(num_sim_state):
            action = self.action_space[0]
            ob, reward, done, info = self.env.step(action)
            state = JointState(self.full_state, ob)
            raw_states.append(state)

        return raw_states

    # gen data from real observation
    def gen_new_data_from_real(self, num_sample, imitation_learning=False, reach_goal=True, add_sim=False):
        for _ in range(num_sample):
            # Get random real state from raw observation
            states = self.get_real_state()
            # Create few imagine state, branching from random real state
            if add_sim:
                states = self.create_imagine_on_real(states)
            # start from end episode (goal or human (collision case)), make up random action for each state, collect reward
            states, rewards = self.edit_episode(states, reach_goal)
            # add data to memory
            self.correct_and_update(states, rewards, imitation_learning)

    # gen data by explore in mix reality
    def gen_data_from_explore_in_mix(self, num_sample, phase="train", min_end=1):
        reach_goal = 0
        collision = 0
        for _ in range(num_sample):
            # get real experience
            raw_states = self.get_real_state(random_epi=True)
            length = random.randrange(min_end, len(raw_states))
            raw_states = raw_states[:length]
            # set env to replay mode
            human_states = raw_states[0].human_states
            self.env.set_current_state(human_states)
            # explore states and collect reward with robot policy
            states = []
            rewards = []
            done = False
            i=0
            joined_state = JointState(self.env.robot.get_full_state(), raw_states[0].human_states)
            info = None
            while not done:
                action = self.policy.predict(joined_state)
                if i+1 < len(raw_states): # replay real experience
                    next_h_action = [h.getvel() for h in raw_states[i+1].human_states]
                else: # imagine next state when needed
                    next_h_action = None
                ob, reward, done, info = self.env.step(action, new_v = next_h_action)
                states.append(self.transform(joined_state))
                rewards.append(reward)
                joined_state = JointState(self.env.robot.get_full_state(), ob)
                i+=1
            self.update_memory(states, rewards)
            if isinstance(info, ReachGoal):
                reach_goal+=1
            if isinstance(info, Collision):
                collision+=1
        success_rate = reach_goal / num_sample
        collision_rate = collision / num_sample
        timeout_rate = (num_sample - reach_goal - collision) / num_sample
        logging.info('Exp in mix has success rate: {:.2f}, collision rate: {:.2f}'
                     .format(success_rate, collision_rate))
        return success_rate, collision_rate, timeout_rate

    def update_memory(self, states, rewards, imitation_learning=False):
        if self.memory is None or self.policy.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                # state = self.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.policy.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.policy.gamma, self.robot.time_step * self.robot.v_pref)
                    value = reward + gamma_bar * self.target_model(next_state.unsqueeze(0)).data.item()
            value = torch.Tensor([value]).to(self.policy.device)
            self.memory.push((state, value))

    def transform(self, state, cadrl=False):
        if cadrl:
            return self.transform_cadrl(state)
        else:
            return self.transform_sarl(state)

    def transform_sarl(self, state):
        """
        Take the state passed from agent and transform it to the input of value network

        :param state:
        :return: tensor of shape (# of humans, len(state))
        """
        state_tensor = torch.cat([torch.Tensor([state.self_state + human_state]).to(self.policy.device)
                                  for human_state in state.human_states], dim=0)
        if self.policy.with_om:
            occupancy_maps = self.policy.build_occupancy_maps(state.human_states)
            state_tensor = torch.cat([self.rotate(state_tensor), occupancy_maps.to(self.policy.device)], dim=1)
        else:
            state_tensor = self.rotate(state_tensor)
        return state_tensor

    def transform_cadrl(self, state):
        """
        Take the state passed from agent and transform it to tensor for batch training

        :param state:
        :return: tensor of shape (len(state), )
        """
        assert len(state.human_states) == 1
        state = torch.Tensor(state.self_state + state.human_states[0]).to(self.policy.device)
        state = self.rotate(state.unsqueeze(0)).squeeze(dim=0)
        return state

    def rotate(self, state):
        """
        Transform the coordinate to agent-centric.
        Input state tensor is of size (batch_size, state_length)

        """
        # 'px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta', 'px1', 'py1', 'vx1', 'vy1', 'radius1'
        #  0     1      2     3      4        5     6      7         8       9     10      11     12       13
        batch = state.shape[0]
        dx = (state[:, 5] - state[:, 0]).reshape((batch, -1))
        dy = (state[:, 6] - state[:, 1]).reshape((batch, -1))
        rot = torch.atan2(state[:, 6] - state[:, 1], state[:, 5] - state[:, 0])

        dg = torch.norm(torch.cat([dx, dy], dim=1), 2, dim=1, keepdim=True)
        v_pref = state[:, 7].reshape((batch, -1))
        vx = (state[:, 2] * torch.cos(rot) + state[:, 3] * torch.sin(rot)).reshape((batch, -1))
        vy = (state[:, 3] * torch.cos(rot) - state[:, 2] * torch.sin(rot)).reshape((batch, -1))

        radius = state[:, 4].reshape((batch, -1))
        if self.policy.kinematics == 'unicycle':
            theta = (state[:, 8] - rot).reshape((batch, -1))
        else:
            # set theta to be zero since it's not used
            theta = torch.zeros_like(v_pref)
        vx1 = (state[:, 11] * torch.cos(rot) + state[:, 12] * torch.sin(rot)).reshape((batch, -1))
        vy1 = (state[:, 12] * torch.cos(rot) - state[:, 11] * torch.sin(rot)).reshape((batch, -1))
        px1 = (state[:, 9] - state[:, 0]) * torch.cos(rot) + (state[:, 10] - state[:, 1]) * torch.sin(rot)
        px1 = px1.reshape((batch, -1))
        py1 = (state[:, 10] - state[:, 1]) * torch.cos(rot) - (state[:, 9] - state[:, 0]) * torch.sin(rot)
        py1 = py1.reshape((batch, -1))
        radius1 = state[:, 13].reshape((batch, -1))
        radius_sum = radius + radius1
        da = torch.norm(torch.cat([(state[:, 0] - state[:, 9]).reshape((batch, -1)), (state[:, 1] - state[:, 10]).
                                  reshape((batch, -1))], dim=1), 2, dim=1, keepdim=True)
        new_state = torch.cat([dg, v_pref, theta, radius, vx, vy, px1, py1, vx1, vy1, radius1, da, radius_sum], dim=1)
        return new_state
