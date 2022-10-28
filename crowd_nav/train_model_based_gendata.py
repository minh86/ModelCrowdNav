#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


parser = argparse.ArgumentParser('Parse configuration file')
parser.add_argument('--env_config', type=str, default='configs_test/env.config')
parser.add_argument('--policy', type=str, default='sarl')
parser.add_argument('--policy_config', type=str, default='configs_test/policy.config')
parser.add_argument('--train_config', type=str, default='configs_test/train.config')
parser.add_argument('--output_dir', type=str, default='data/sarl5')
parser.add_argument('--weights', type=str)
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--gpu', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--add_noise', default=True, action='store_true')

args = parser.parse_args()

# In[3]:


# configure paths
make_new_dir = True
if os.path.exists(args.output_dir):
    key = input('Output directory already exists! Overwrite the folder? (y/n)')
    if key == 'y' and not args.resume:
        shutil.rmtree(args.output_dir)
    else:
        make_new_dir = False
        args.env_config = os.path.join(args.output_dir, os.path.basename(args.env_config))
        args.policy_config = os.path.join(args.output_dir, os.path.basename(args.policy_config))
        args.train_config = os.path.join(args.output_dir, os.path.basename(args.train_config))
if make_new_dir:
    os.makedirs(args.output_dir)
    shutil.copy(args.env_config, args.output_dir)
    shutil.copy(args.policy_config, args.output_dir)
    shutil.copy(args.train_config, args.output_dir)
log_file = os.path.join(args.output_dir, 'output.log')
il_weight_file = os.path.join(args.output_dir, 'il_model.pth')
rl_weight_file = os.path.join(args.output_dir, 'rl_model.pth')

# configure logging
mode = 'a' if args.resume else 'w'
file_handler = logging.FileHandler(log_file, mode=mode)
stdout_handler = logging.StreamHandler(sys.stdout)
level = logging.INFO if not args.debug else logging.DEBUG
logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                    format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
# repo = git.Repo(search_parent_directories=True)
# logging.info('Current git head hash code: %s'.format(repo.head.object.hexsha))
# device = torch.device("cuda:0" if torch.cuda.is_available() and args.gpu else "cpu")
device = torch.device(args.device)
logging.info('Using device: %s', device)
mem_path = os.path.join(args.output_dir, 'memory.data')
rawob_path = os.path.join(args.output_dir, 'rawob.data')
model_sim_checkpoint = os.path.join(args.output_dir, 'model_sim.pt')
# configure policy
policy = policy_factory[args.policy]()
if not policy.trainable:
    parser.error('Policy has to be trainable')
if args.policy_config is None:
    parser.error('Policy config has to be specified for a trainable network')
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
env.set_robot(robot)

# read training parameters
if args.train_config is None:
    parser.error('Train config has to be specified for a trainable network')
train_config = configparser.RawConfigParser()
train_config.read(args.train_config)
rl_learning_rate = train_config.getfloat('train', 'rl_learning_rate')
train_batches = train_config.getint('train', 'train_batches')
train_episodes = train_config.getint('train', 'train_episodes')
sample_episodes = train_config.getint('train', 'sample_episodes')
target_update_interval = train_config.getint('train', 'target_update_interval')
evaluation_interval = train_config.getint('train', 'evaluation_interval')
capacity = train_config.getint('train', 'capacity')
epsilon_start = train_config.getfloat('train', 'epsilon_start')
epsilon_end = train_config.getfloat('train', 'epsilon_end')
epsilon_decay = train_config.getfloat('train', 'epsilon_decay')
checkpoint_interval = train_config.getint('train', 'checkpoint_interval')

init_episodes = train_config.getint('train_sim', 'init_episodes')
model_sim_lr = train_config.getfloat('train_sim', 'model_sim_lr')
model_sim_epochs = train_config.getint('train_sim', 'model_sim_epochs')
sample_episodes_in_real = train_config.getint('train_sim', 'sample_episodes_in_real')
ms_batchsize = train_config.getint('train_sim', 'ms_batchsize')
sample_episodes_in_sim = train_config.getint('train_sim', 'sample_episodes_in_sim')

# configure trainer and explorer
memory = ReplayMemory(capacity)
model = policy.get_model()
batch_size = train_config.getint('trainer', 'batch_size')
trainer = Trainer(model, memory, device, batch_size)
explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)
explorer.rawob = ReplayMemory(capacity)

# config sim environment
model_sim = mlp(env_config.getint('sim', 'human_num'));
model_sim.to(device)
model_sim.device = device
env_sim = gym.make('ModelCrowdSim-v0')
env_sim.configure(env_config)
env_sim.set_robot(robot)
env_sim.device = device
env_sim.sim_world = model_sim
env.device = device
env.sim_world = model_sim
env_sim.add_noise = args.add_noise
# model based things
trainer_sim = Trainer_Sim(model_sim, explorer.rawob, device, ms_batchsize, model_sim_checkpoint)
explorer_sim = Explorer(env_sim, robot, device, memory, policy.gamma, target_policy=policy)
# datagen things
robot.set_policy(policy)

# reinforcement learning
policy.set_env(env)
trainer_sim.set_learning_rate(model_sim_lr)
robot.print_info()
trainer.set_learning_rate(rl_learning_rate)
data_generator = DataGen(memory, robot, env_sim)
data_generator.update_target_model(model)
episode = 0
epsilon = epsilon_end  # fix small epsilon
robot.policy.set_epsilon(epsilon)

# explore real to train sim
explorer.run_k_episodes(sample_episodes_in_real, 'train', update_memory=False, update_raw_ob=True, stay=True)
ms_valid_loss = trainer_sim.optimize_epoch(model_sim_epochs)
# logging.info('Model-based env.  val_loss: {:.4f}'.format(ms_valid_loss))

for episode in tqdm(range(train_episodes)):
    # evaluate the model
    if episode % evaluation_interval == 0 and episode != 0:
        logging.info("Val in real...")
        explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)

    # gen sim data and train
    data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=True)
    data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=False)
    average_loss = trainer.optimize_batch(train_batches)
    # logging.info('Policy model env. val_loss: {:.4f}'.format(average_loss))

    if episode % target_update_interval == 0:
        data_generator.update_target_model(model)

    if episode != 0 and episode % checkpoint_interval == 0:
        torch.save(model.state_dict(), rl_weight_file)

# final test
explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)
