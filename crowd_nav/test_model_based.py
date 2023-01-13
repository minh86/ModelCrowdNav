import gc
import logging
import argparse
import configparser
import os

import neptune.new as neptune
import torch
import gym
import sys


sys.path.append('../')
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.explorer import Explorer
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.utils.misc import *
from crowd_nav.utils.datagen import DataGen

def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--input_file', type=str, default='data/sarl5/list.txt')
    parser.add_argument('--output_file', type=str, default='data/sarl5/test.txt')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_dataset', default=False, action='store_true')  # using dataset instead of simulator
    parser.add_argument('--replace_robot', default=False, action='store_true')  # replace human as robot
    parser.add_argument('--cutting_point', type=int, default=-1)  # split point for train_val and test dataset

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    data = []
    env = None
    env_sim = None
    # read input file: input_dir \t video_tag
    out_file = open(args.output_file, "w")
    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            data.append(line)
    for case in data:
        logging.info("================= WORKING IN %s =================" % case[0])
        gc.collect()
        input_dir = case[0]
        # configure paths
        env_config_file = os.path.join(input_dir, "env.config")
        policy_config_file = os.path.join(input_dir, "policy.config")
        train_config_file = os.path.join(input_dir, "train.config")

        device = torch.device(args.device)
        logging.info('Using device: %s', device)

        # configure policy
        policy = policy_factory['sarl']()
        policy_config = configparser.RawConfigParser()
        policy_config.read(policy_config_file)
        policy.configure(policy_config)
        policy.set_device(device)

        # configure environment
        env_config = configparser.RawConfigParser()
        env_config.read(env_config_file)
        if env is None:
            env = gym.make('CrowdSim-v0')
            env.configure(env_config)
        robot = Robot(env_config, 'robot')
        robot.set_policy(policy)
        env.set_robot(robot)
        explorer = Explorer(env, robot, device, gamma=0.9)

        # config env_sim
        if env_sim is None:
            env_sim = gym.make('ModelCrowdSim-v0')
            env_sim.configure(env_config)
        if not policy.multiagent_training:
            env_sim.human_num = 1
        env_sim.set_robot(robot)
        env_sim.device = device
        explorer_sim = Explorer(env_sim, robot, device, None, policy.gamma, target_policy=policy)

        # read training parameters
        if train_config_file is None:
            parser.error('Train config has to be specified for a trainable network')
        train_config = configparser.RawConfigParser()
        train_config.read(train_config_file)

        rl_weight_file = os.path.join(input_dir, 'rl_model.pth')
        last_rl_weight_file = os.path.join(input_dir, 'last_rl_model.pth')

        if os.path.exists(last_rl_weight_file):
            w_file = last_rl_weight_file
        else:
            w_file = rl_weight_file

        logging.info("Loading model file: %s" % w_file)
        robot.policy.model.load_state_dict(torch.load(w_file, map_location=device))  # load best model
        if args.use_dataset:
            test_datapath = train_config.get('dataset', 'test_datapath')
            stride = train_config.getint('dataset', 'stride')
            windows_size = train_config.getint('dataset', 'windows_size')
            view_distance = train_config.getint('train_sim', 'view_distance')
            view_human = train_config.getint('train_sim', 'view_human')

            test_sl = None
            if args.cutting_point > 0:
                test_sl = [args.cutting_point, env.case_size['test']]

            data_generator = DataGen(None, robot, env_sim, policy)
            test_raw_memory, _ = GetRealData(dataset_file=test_datapath, phase="test", stride=stride,
                                             windows_size=windows_size, dataset_slice=test_sl, Store_for_world_fn=StoreAction)
            data_generator.raw_memory = test_raw_memory
            data_generator.counter = 0
            cumulative_rewards, success_rate, collision_rate, timeout_rate, nav_time = data_generator.gen_data_from_explore_in_mix(
                env.case_size['test'],
                random_robot=False,
                add_sim=False,
                random_epi=False,
                phase='test',
                view_distance=view_distance,
                view_human=view_human,
                returnRate=True,
                updateMemory=False,
                replace_robot=args.replace_robot,
                returnNav=True
            )
            out_file.write("--- %s ---\treward: %s\tsuccess rate: %s\tcollision_rate:%s\ttimeout rate:%s\tnavigation "
                           "time:%s\n" %
                           (case[1], cumulative_rewards, success_rate, collision_rate, timeout_rate, nav_time))
        else:
            cumulative_rewards, success_rate, collision_rate, timeout_rate, nav_time = explorer.run_k_episodes(
                env.case_size['test'], 'test', returnNav=True)
            out_file.write("--- %s ---\treward: %s\tsuccess rate: %s\tcollision_rate:%s\ttimeout rate:%s\tnavigation "
                           "time:%s\n" %
                           (case[1], cumulative_rewards, success_rate, collision_rate, timeout_rate, nav_time))

if __name__ == '__main__':
    main()
