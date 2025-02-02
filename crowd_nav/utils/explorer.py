import logging
import copy
import os

import numpy as np
import torch
from tqdm import tqdm

from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.action import ActionRot, ActionXY

class Explorer(object):
    def __init__(self, env, robot, device, memory=None, gamma=None, target_policy=None):
        self.raw_memory = None
        self.env = env
        self.robot = robot
        self.device = device
        self.memory = memory
        self.rawob = None
        self.gamma = gamma
        self.target_policy = target_policy
        self.target_model = None

    def update_target_model(self, target_model):
        self.target_model = copy.deepcopy(target_model)

    @staticmethod
    def someone_is_moving(ob):
        min_speed = 1e-3
        for h in ob:
            if abs(h.vx) > min_speed or abs(h.vy) > min_speed:
                return True
        return False

    # @profile
    def run_k_episodes(self, k, phase, update_memory=False, imitation_learning=False, episode=None,
                       print_failure=False,update_raw_ob=False, stay=False, returnRate=True, test_case=None, returnNav=False,
                       cacheFile=None):
        self.robot.policy.set_phase(phase)
        success_times = []
        collision_times = []
        timeout_times = []
        success = 0
        collision = 0
        timeout = 0
        too_close = 0
        min_dist = []
        cumulative_rewards = []
        collision_cases = []
        timeout_cases = []
        avg_nav_time = np.NaN
        fcount = 0
        for i in tqdm(range(k)):
            ob = self.env.reset(phase, test_case=test_case)
            done = False
            states = []
            actions = []
            rewards = []
            current_s = None
            cacheFrames = []
            frameid = 0
            while not done:
                if stay:
                    holonomic = True if self.robot.policy.kinematics == 'holonomic' else False
                    action = ActionXY(0, 0) if holonomic else ActionRot(0, 0)
                else:
                    action = self.robot.act(ob)
                current_s = [tmpo.getvalue() for tmpo in ob]
                ob, reward, done, info = self.env.step(action)
                next_s = [tmpo.getvalue() for tmpo in ob]
                next_action = [s[2:] for s in next_s]
                next_action = next_action[:len(current_s)]

                # Store observation for sgan dataloader
                frameid += 10
                for p_id, tmp_pes in enumerate(ob):
                    cacheFrames.append([frameid, p_id, tmp_pes.px, tmp_pes.py])

                # Store observation for generating data
                if self.raw_memory is not None:
                    self.raw_memory.push((ob,reward,done,info))

                # Create training data for model-based
                if update_raw_ob: # State , Next human's action
                    if self.someone_is_moving(ob):
                        self.rawob.push((torch.Tensor(current_s), torch.Tensor(next_action)))
                
                states.append(self.robot.policy.last_state)
                actions.append(action)
                rewards.append(reward)

                if isinstance(info, Danger):
                    too_close += 1
                    min_dist.append(info.min_dist)

            if isinstance(info, ReachGoal):
                success += 1
                success_times.append(self.env.global_time)
            elif isinstance(info, Collision):
                collision += 1
                collision_cases.append(i)
                collision_times.append(self.env.global_time)
            elif isinstance(info, Timeout):
                timeout += 1
                timeout_cases.append(i)
                timeout_times.append(self.env.time_limit)
            else:
                raise ValueError('Invalid end signal from environment')

            if update_memory:
                if isinstance(info, ReachGoal) or isinstance(info, Collision):
                    # only add positive(success) or negative(collision) experience in experience set
                    self.update_memory(states, actions, rewards, imitation_learning)

            # Save to file for sgan
            if cacheFile is not None:
                fcount += 1
                with open(os.path.join(cacheFile, str(fcount) + ".txt"), 'w') as cache_file_h:
                    for frame in cacheFrames:
                        cache_file_h.write(
                            "%s\t%s\t%s\t%s\n" % (frame[0], frame[1], frame[2], frame[3]))


            cumulative_rewards.append(sum([pow(self.gamma, t * self.robot.time_step * self.robot.v_pref)
                                           * reward for t, reward in enumerate(rewards)]))

        success_rate = success / k
        collision_rate = collision / k
        timeout_rate = (k - success - collision) / k
        assert success + collision + timeout == k
        avg_nav_time = sum(success_times) / len(success_times) if success_times else self.env.time_limit

        extra_info = '' if episode is None else 'in episode {} '.format(episode)
        if stay == False:
            logging.info('{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, nav time: {:.2f}, total reward: {:.4f}'.
                     format(phase.upper(), extra_info, success_rate, collision_rate, avg_nav_time,
                            average(cumulative_rewards)))
        if phase in ['val', 'test']:
            num_step = sum(success_times + collision_times + timeout_times) / self.robot.time_step
            logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                         too_close / num_step, average(min_dist))

        if print_failure:
            logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
            logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))
        if returnRate and returnNav:
            return average(cumulative_rewards), success_rate, collision_rate, timeout_rate, avg_nav_time
        if returnRate:
            return average(cumulative_rewards), success_rate, collision_rate, timeout_rate
        else:
            return average(cumulative_rewards), success, collision, (k - success - collision)

    def update_memory(self, states, actions, rewards, imitation_learning=False):
        if self.memory is None or self.gamma is None:
            raise ValueError('Memory or gamma value is not set!')

        for i, state in enumerate(states):
            reward = rewards[i]

            # VALUE UPDATE
            if imitation_learning:
                # define the value of states in IL as cumulative discounted rewards, which is the same in RL
                state = self.target_policy.transform(state)
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
            value = torch.Tensor([value]).to(self.device)

            # # transform state of different human_num into fixed-size tensor
            # if len(state.size()) == 1:
            #     human_num = 1
            #     feature_size = state.size()[0]
            # else:
            #     human_num, feature_size = state.size()
            # if human_num != 5:
            #     padding = torch.zeros((5 - human_num, feature_size))
            #     state = torch.cat([state, padding])
            self.memory.push((state, value))


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
