import gc
import logging
import argparse
import configparser
import os

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
    parser.add_argument('--output_dir', type=str, default='data/sarl5')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--use_dataset', default=False, action='store_true')  # using dataset instead of simulator
    parser.add_argument('--replace_robot', default=False, action='store_true')  # replace human as robot
    parser.add_argument('--cutting_point', type=int, default=-1)  # split point for train_val and test dataset
    parser.add_argument('--test_case', type=int, default=1)  # test case number
    parser.add_argument('--human_num', type=int, default=-1)

    args = parser.parse_args()

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    data = []
    env = None
    env_sim = None
    # read input file: input_dir \t video_tag
    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip().split()
            data.append(line)
    for case in data:
        logging.info("================= WORKING IN %s =================" % case[0])
        gc.collect()
        input_dir = case[0]
        video_tag = case[1]
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
        weight_files = [rl_weight_file, last_rl_weight_file]

        for w_file in weight_files:
            if not os.path.exists(w_file):
                continue
            logging.info("Loading model file: %s" % w_file)
            robot.policy.model.load_state_dict(torch.load(w_file, map_location=device))  # load best model
            f = os.path.basename(w_file).split('.')[0]
            if args.use_dataset:
                # env.human_num = args.human_num
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
                cumulative_rewards, success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(
                    1,  # test size for crowds_students003.ndjson
                    # max_human=max_human,
                    random_robot=False,
                    add_sim=False,
                    random_epi=False,
                    phase='test',
                    # render_path=args.output_dir,
                    view_distance=view_distance,
                    view_human=view_human,
                    returnRate=True,
                    updateMemory=False,
                    test_case=args.test_case,
                    replace_robot=args.replace_robot,
                )
                traj_file = os.path.join(args.output_dir, "%s_%s_%s_%s.pdf" % (f, video_tag, str(args.test_case),str(int(success_rate))))
                explorer_sim.env.render("traj", traj_file)
                logging.info("Saved traj plot at: %s" % traj_file)
            else:
                cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(
                    1, 'test', test_case=args.test_case)
                traj_file = os.path.join(args.output_dir,
                                         "%s_%s_%s_%s.pdf" % (f, video_tag, str(args.test_case), str(int(success_rate))))
                explorer.env.render("traj", traj_file)
                logging.info("Saved traj plot at: %s" % traj_file)

if __name__ == '__main__':
    main()
