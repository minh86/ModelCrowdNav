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


def main():
    parser = argparse.ArgumentParser('Parse configuration file')
    parser.add_argument('--env_config', type=str, default='configs_test/env.config')
    parser.add_argument('--policy', type=str, default='sarl')
    parser.add_argument('--policy_config', type=str, default='configs_test/policy.config')
    parser.add_argument('--train_config', type=str, default='configs_test/train.config')
    parser.add_argument('--output_dir', type=str, default='data/sarl5')
    parser.add_argument('--weights', type=str)
    parser.add_argument('--resume', default=False, action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--neptune', default=False, action='store_true')
    parser.add_argument('--neptune_name', type=str, default='Untitled')

    parser.add_argument('--min_human_num', type=int, default=5)
    parser.add_argument('--max_human_num', type=int, default=11)
    parser.add_argument('--step_human_num', type=int, default=2)

    args = parser.parse_args()

    # configure paths
    args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
    args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
    args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))

    rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')
    last_rl_weight_file = os.path.join(args.output_dir, 'last_rl_model.pth')
    weight_files = [rl_weight_file]
    if os.path.exists(last_rl_weight_file):
        weight_files.append(last_rl_weight_file)

    # configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device(args.device)
    logging.info('Using device: %s', device)

    # configure policy
    policy = policy_factory[args.policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(args.policy_config)
    policy.configure(policy_config)
    policy.set_device(device)

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(args.env_config)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    env.set_robot(robot)
    explorer = Explorer(env, robot, device, gamma=0.9)

    # read training parameters
    if args.train_config is None:
        parser.error('Train config has to be specified for a trainable network')
    train_config = configparser.RawConfigParser()
    train_config.read(args.train_config)
    api_token = train_config.get('neptune', 'api_token')
    neptune_project = train_config.get('neptune', 'neptune_project')

    # ----------------------------  neptune params --------------------------------
    params = {"output_dir": args.output_dir, "min_human_num": args.min_human_num,
              "max_human_num": args.max_human_num, "step_human_num": args.step_human_num,
              "device": args.device
              }

    # ============== neptune things  ================
    if args.neptune:
        run = neptune.init_run(
            project=neptune_project,
            api_token=api_token,
            name=args.neptune_name
        )
        run["parameters"] = params
        run["config/env"].upload(args.env_config)
        run["config/policy"].upload(args.policy_config)
        run["config/train"].upload(args.train_config)
    for w_file in weight_files:
        robot.policy.model.load_state_dict(torch.load(w_file))  # load best model
        f = os.path.basename(w_file).split('.')[0]
        for i in range(args.min_human_num, args.max_human_num, args.step_human_num):
            # final test
            env.human_num = i
            cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(
                env.case_size['test'], 'test')
            video_tag = "test_vi"
            video_file = os.path.join(args.output_dir, "%s_%s_%d.gif" %(video_tag,f,i) )
            explorer.env.render("video", video_file)
            if args.neptune:
                run["test_%s/success_rate" % f].log(success_rate)  # log to neptune
                run["test_%s/collision_rate" % f].log(collision_rate)  # log to neptune
                run["test_%s/timeout_rate" % f].log(timeout_rate)  # log to neptune
                run["test_%s/human_num" % f].log(i)  # log to neptune
                Resize_GIF(video_file)
                run[video_tag + "/" + "%s_%d.gif" %(f,i)].upload(video_file)  # upload to neptune
    if args.neptune:
        run.stop()


if __name__ == '__main__':
    main()
