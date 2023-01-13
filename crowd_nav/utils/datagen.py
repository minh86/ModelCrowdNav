import logging
import itertools
import copy
import math
import os
import random
from collections import namedtuple

import numpy as np
from numpy.linalg import norm
import torch
from tqdm import tqdm

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.state import JointState, FullState
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.utils import point_to_segment_dist


class DataGen(object):

    def __init__(self, memory, robot, env, policy):
        self.counter = 0
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
        self.gamma = policy.gamma

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    def gen_new_episode(self, min_epi_length=30, max_epi_length=60, phase="train", max_human=-1):
        raw_states = []
        done = False
        i = 0
        if max_human > 0:
            self.env.human_num = max_human
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
    def gen_new_data(self, num_sample, imitation_learning=False, reach_goal=True, max_human=-1):
        for _ in range(num_sample):
            # Create random episode as robot stay the same position
            states = self.gen_new_episode(max_human=max_human)
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

    # count episodes
    def count(self):
        return len(self.get_episode_start_index())

    # get index for first state
    def get_episode_start_index(self):
        indexes = [];
        add = True
        for i, data in enumerate(self.raw_memory):
            if add:
                indexes.append(i)
                add = False
            if data[2]: # done
                add = True
        return indexes

    # get a real episode
    def pick_real_episode(self, random_epi=False, max_human=-1, test_case=None):
        obs = []
        indexes = self.get_episode_start_index()
        if random_epi:
            i = random.choice(indexes)
        else:
            i = indexes[self.counter % len(indexes)]
            self.counter += 1
        if test_case is not None:
            i = test_case
        mem_states = self.raw_memory[i:]
        for i, data in enumerate(mem_states):
            # data: ob, reward, done, info, start_ends
            # if self.someone_is_moving(ob):
            ob = data[0]
            done = data[2]
            if 0 < max_human < len(ob):
                ob = ob[:max_human]
            obs.append(ob)
            if done:
                break
        if len(data) == 5 :# has start_ends
            start_ends = data[4]
            if max_human>0:
                return obs, start_ends[:max_human]
            else:
                return obs, start_ends
        else:
            return obs, None

    # create JointState from last real episode
    def get_real_state(self, random_epi=False, max_human=-1, random_robot=True, replace_robot=False, test_case=None):
        RobotInfo = namedtuple('RobotInfo', ['px', 'py', 'gx', 'gy'])
        px, py, gx, gy = None, None, None, None
        obs, start_end = self.pick_real_episode(random_epi=random_epi, max_human=max_human, test_case=test_case)
        raw_states = []
        robot_info = None
        if replace_robot: #Replace robot only
            distances = [np.linalg.norm([p[2] - p[0], p[3] - p[1]]) for p in start_end]
            avr_dis = np.average(distances)
            possible_case = [i for i in range(len(distances)) if (self.env.time_limit * self.robot.v_pref)
                             > distances[i] > avr_dis]
            if random_robot is False:  # replace robot by human with the possible longest path
                i_poss = list(enumerate(distances))
                sorted_poss = sorted(i_poss, key=lambda x: x[1])[-len(possible_case):][::-1]
                possible_case = [case[0] for case in sorted_poss]
                min_dis = 0
                while min_dis < self.robot.radius * 4:  # check if robot collide with human at init state
                    if len(possible_case) == 0:  # cant find possible init position
                        return [], RobotInfo(None, None, None, None)
                    set_robot = possible_case.pop(0)
                    init_state = obs[0][:set_robot] + obs[0][set_robot + 1:]
                    # pad start, end position
                    [px, py, gx, gy] = [start_end[set_robot][i] for i in range(4)]
                    moving_vector = [gx - px, gy - py]
                    padding_dis = 2
                    pad_x = padding_dis * math.sin(moving_vector[0] / np.linalg.norm(moving_vector))
                    pad_y = padding_dis * math.sin(moving_vector[1] / np.linalg.norm(moving_vector))
                    px, py, gx, gy = px - pad_x, py - pad_y, gx + pad_x, gy + pad_y
                    init_dis = [np.linalg.norm([px-h.px, py-h.py]) for h in init_state]
                    if len(init_dis) >0:
                        min_dis = min(init_dis)
            else:
                # random replace human with robot
                min_dis = 0
                while min_dis < self.robot.radius*4: # check if robot collide with human at init state
                    if len(possible_case) == 0:# cant find possible init position
                        return [], RobotInfo(None, None, None, None)
                    set_robot = possible_case.pop(random.randrange(len(possible_case)))
                    init_state = obs[0][:set_robot]+obs[0][set_robot+1:]
                    # pad start, end position
                    [px, py, gx, gy] = [start_end[set_robot][i] for i in range(4)]
                    moving_vector = [gx-px, gy-py]
                    padding_dis = 2
                    pad_x = padding_dis*math.sin(moving_vector[0]/np.linalg.norm(moving_vector))
                    pad_y = padding_dis*math.sin(moving_vector[1]/np.linalg.norm(moving_vector))
                    px, py, gx, gy = px-pad_x, py-pad_y, gx+pad_x, gy+pad_y

                    init_dis = [np.linalg.norm([px - h.px, py - h.py]) for h in init_state]
                    if len(init_dis) > 0:
                        min_dis = min(init_dis)

            # set start and goal position for robot
            robot_info = RobotInfo(px, py, gx, gy)

            # remove human traj which is replaced by robot
            h_id = set_robot
            obs = [ob[:h_id] + ob[h_id + 1:] if len(ob) > h_id else ob for ob in obs]

        for ob in obs:
            state = JointState(self.full_state, ob)
            raw_states.append(state)
        return raw_states, robot_info

    # add sim states to real states
    def create_imagine_on_real(self, states, min_end=7, min_sim_state=3, max_sim_state=15):
        length = random.randrange(min_end, len(states) - 5)
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

    # keep human state in range of view_distance only
    def CorrectViewByDistance(self, joined_state, view_distance):
        selfState, humanState = joined_state.self_state, joined_state.human_states
        rpx, rpy = selfState.px, selfState.py
        distance = [np.linalg.norm([rpx - h.px, rpy - h.py]) for h in humanState]
        closest = np.argmin(distance)
        valid_human = [i for i in range(len(distance)) if distance[i] <= view_distance]
        if len(valid_human) == 0: # pad with the closest human if no one in view range
            valid_human = [closest]
        humanState = [humanState[i] for i in valid_human]
        joined_state = JointState(selfState, humanState)
        return joined_state

    # keep n the closest humans state only
    def CorrectViewByNHuman(self, joined_state, human_count):
        selfState, humanState = joined_state.self_state, joined_state.human_states
        rpx, rpy = selfState.px, selfState.py
        distance = [np.linalg.norm([rpx - h.px, rpy - h.py]) for h in humanState]
        i_distance = list(enumerate(distance))
        n_closest = sorted(i_distance, key=lambda x: x[1])[:human_count]
        valid_human = [h[0] for h in n_closest]
        closest = np.argmin(distance)
        if len(valid_human) == 0:  # pad with the closest human if no one in view range
            valid_human = [closest]
        humanState = [humanState[i] for i in valid_human]
        joined_state = JointState(selfState, humanState)
        return joined_state

    # gen data by explore in mix reality
    def gen_data_from_explore_in_mix(self, num_sample, phase="train", min_end=1, static_end=-1, max_human=-1, imitation_learning=False,
                                     add_sim=True, stay=False, random_epi=True, random_robot=True, render_path=None,
                                     view_distance=-1, view_human=-1, returnRate=False, updateMemory=True, replace_robot=False,
                                     sgan_genfile=None, test_case=None):
        '''
        min_end: the minimum length of real data
        max_end: the static length of real data
        render_path: render every episode (for debug only)
        random_robot: replace robot start end goal by random human traj
        random_epi: pick random episode from real experience
        add_sim: add sim state from world model
        sgan_genfile: input file for generating trajectories
        test_case: test case id
        '''

        reach_goal = 0
        collision = 0
        cumulative_rewards = []
        success_times = []
        collision_times = []
        timeout_times = []
        too_close = 0
        min_dist = []
        self.policy.set_phase(phase)
        c_sample = 0
        pbar = None
        if imitation_learning or phase == "val" or phase == "test":
            pbar = tqdm(total=num_sample, position=0, leave=True)
        while c_sample < num_sample:
            # get real experience
            raw_states, robot_info = self.get_real_state(random_epi=random_epi, max_human=max_human,
                                                         random_robot=random_robot, replace_robot=replace_robot,
                                                         test_case=test_case)
            if raw_states == []:
                continue
            if len(raw_states) <= min_end:
                continue
            length = len(raw_states)
            if add_sim:
                length = random.randrange(min_end, len(raw_states))
                if static_end > 0:
                    length = static_end # static length
            raw_states = raw_states[:length]
            # create input for sgan
            if sgan_genfile is not None:
                frameid = 0
                with open(sgan_genfile, 'w') as sgan_genfile_h:
                    tmp = raw_states[-min_end:]
                    for state in tmp:
                        for i, human in enumerate(state.human_states):
                            sgan_genfile_h.write("%s\t%s\t%s\t%s\n" % (frameid, i, human.px, human.py))
                        frameid += 1

            # set env to replay mode
            human_states = raw_states[0].human_states
            self.env.set_current_state(human_states, robot_info)
            # explore states and collect reward with robot policy
            states = []
            rewards = []
            done = False
            i = 0
            joined_state = JointState(self.env.robot.get_full_state(), raw_states[0].human_states)
            info = None
            if pbar is not None:  pbar.update(1)
            while not done:
                if not stay:
                    if view_distance >0 :
                        joined_state = self.CorrectViewByDistance(joined_state, view_distance)
                    if view_human >0 :
                        joined_state = self.CorrectViewByNHuman(joined_state, view_human)
                    action = self.policy.predict(joined_state)
                else:  # make robot stay for debug
                    holonomic = True if self.robot.policy.kinematics == 'holonomic' else False
                    action = ActionXY(0, 0) if holonomic else ActionRot(0, 0)

                # replay real experience
                if i + 1 < len(raw_states):
                    next_h_action = [h.getvel() for h in raw_states[i + 1].human_states][:self.env.human_num]

                    ob, reward, done, info = self.env.step(action, new_v=next_h_action)
                    # add more human when needed
                    n_state = raw_states[i + 1]
                    n_human_num = len(n_state.human_states)
                    if n_human_num > self.env.human_num:
                        diff = n_human_num - self.env.human_num
                        self.env.humans += [copy.deepcopy(self.env.humans[0]) for _ in range(diff)]
                        for h_i, h_state in enumerate(n_state.human_states):
                            self.env.humans[h_i].set(h_state.px, h_state.py, 0, 0, h_state.vx, h_state.vy, 0)
                        self.env.human_num = n_human_num
                        ob = n_state.human_states

                # imagine next state when needed
                else:
                    if add_sim:
                        next_h_action = None # add imagine state from world model
                    else:
                        next_h_action = [[0, 0]] * self.env.human_num # humans stop moving
                    ob, reward, done, info = self.env.step(action, new_v=next_h_action)

                states.append(self.transform(joined_state))
                rewards.append(reward)
                joined_state = JointState(self.env.robot.get_full_state(), ob)
                i += 1
                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal) or isinstance(info, Collision):
                if updateMemory:
                    self.update_memory(states, rewards, imitation_learning=imitation_learning)
            if isinstance(info, ReachGoal):
                reach_goal += 1
                success_times.append(self.env.global_time)
            if isinstance(info, Collision):
                collision += 1
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout_times.append(self.env.time_limit)
            if render_path is not None:
                self.env.render("video", os.path.join(render_path, str(c_sample) + ".gif"))
            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))
            c_sample += 1
        if pbar is not None:
            pbar.close()
        success_rate = reach_goal / num_sample
        collision_rate = collision / num_sample
        timeout_rate = (num_sample - reach_goal - collision) / num_sample
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit
        logging.info('Exp in mix has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'
                     .format(success_rate, collision_rate, avg_nav_time, np.average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, np.average(min_dist))
        if returnRate:
            return np.average(cumulative_rewards), success_rate, collision_rate, timeout_rate
        return np.average(cumulative_rewards), reach_goal, collision, (num_sample - reach_goal - collision)

    def update_memory(self, states, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                # state = self.transform(state)
                # value = pow(self.gamma, (len(states) - 1 - i) * self.robot.time_step * self.robot.v_pref)
                value = sum([pow(self.gamma, max(t - i, 0) * self.robot.time_step * self.robot.v_pref) * reward
                             * (1 if t >= i else 0) for t, reward in enumerate(rewards)])
            else:
                if i == len(states) - 1:
                    # terminal state
                    value = reward
                else:
                    next_state = states[i + 1]
                    gamma_bar = pow(self.gamma, self.robot.time_step * self.robot.v_pref)
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
        if hasattr(self.policy, 'with_om') and self.policy.with_om:
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
