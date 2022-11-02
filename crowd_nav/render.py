#!/usr/bin/env python
# coding: utf-8

import sys
import logging
import argparse
import configparser
import os
import shutil
import torch
import gym
import sys, os, pickle
from tqdm import tqdm

sys.path.append('../')
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.trainer_sim import Trainer_Sim
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.datagen import DataGen
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.world_model import *

parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--env_config', type=str, default='configs_test/env.config')
parser.add_argument('--policy', type=str, default='sarl')
parser.add_argument('--policy_config', type=str, default='configs_test/policy.config')
parser.add_argument('--output_dir', type=str, default='data/sarl5')
parser.add_argument('--weights', type=str)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--device', type=str, default='cpu')


args = parser.parse_args()

def config_robot(policy, model_weight, env, robot, device):
    policy_config_file = "configs/policy.config"
    # configure policy
    policy = policy_factory[policy]()
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    policy.set_phase("test")
    policy.set_device(device)
    policy.set_env(env)
    policy.get_model().load_state_dict(torch.load(model_weight, map_location=device))
    robot.set_policy(policy)
    robot.policy.set_epsilon(0)
    env.set_robot(robot)

def doTestRender(startcase, endcase, explorer, output_dir, video_tag, device=None, meta=False, k_shot=5,
                 n_way=1, capacity=5000, batch_size=100, lr=0.01, train_epsilon=0.1):
    explorer.gamma = explorer.robot.policy.gamma

    for test_case in range(startcase, endcase):
        explorer.run_k_episodes(1, "test", print_failure=True, test_case=test_case)
        explorer.env.render("video", os.path.join(output_dir, video_tag + "_c" + str(test_case) + ".gif"))


env_config_file = args.env_config
num_case = 100

logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                    datefmt="%Y-%m-%d %H:%M:%S")
device = torch.device(args.device)

env_config = configparser.RawConfigParser()
env_config.read(env_config_file)
env = gym.make('CrowdSim-v0')
env.configure(env_config)
robot = Robot(env_config, 'robot')
explorer = Explorer(env, robot, device, gamma=0.9)
startcase=501
endcase = 502
output_dir="data/videos"

config_robot(args.policy,args.model, env, robot,device)
doTestRender(startcase, endcase, explorer, output_dir, "del_me")