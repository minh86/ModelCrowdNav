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
sys.path.append('../')
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.trainer_sim import Trainer_Sim
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
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
parser.add_argument('--sim_only', default=False, action='store_true')

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
device = torch.device(args.device )
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
model_sim = autoencoder(env_config.getint('sim', 'human_num')); model_sim.to(device)
env_sim = gym.make('ModelCrowdSim-v0')
env_sim.configure(env_config)
env_sim.set_robot(robot)
env_sim.device = device
env_sim.sim_world = model_sim
env.device = device
env.sim_world = model_sim
# model based things
trainer_sim = Trainer_Sim(model_sim, explorer.rawob, device, ms_batchsize, model_sim_checkpoint)
explorer_sim = Explorer(env_sim, robot, device, memory, policy.gamma, target_policy=policy)
sim_only = args.sim_only

# In[4]:


# Sample data for model training
il_episodes = train_config.getint('imitation_learning', 'il_episodes')
il_policy = train_config.get('imitation_learning', 'il_policy')
il_epochs = train_config.getint('imitation_learning', 'il_epochs')
il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
if robot.visible:
    safety_space = 0
else:
    safety_space = train_config.getfloat('imitation_learning', 'safety_space')
il_policy = policy_factory[il_policy]()
il_policy.multiagent_training = policy.multiagent_training
il_policy.safety_space = safety_space
robot.set_policy(il_policy)

# sample data from real env
explorer.run_k_episodes(il_episodes, 'train', update_memory=False, imitation_learning=True,update_raw_ob=True)

# Saving memory
logging.info("Saving memory: %s",mem_path)
with open(mem_path, 'wb') as f:
    pickle.dump(memory,f)
logging.info("Saving raw observation: %s",rawob_path)
with open(rawob_path, 'wb') as f:
    pickle.dump(explorer.rawob,f)
    
# training sim model
trainer_sim.set_learning_rate(model_sim_lr)
ms_valid_loss = trainer_sim.optimize_epoch(model_sim_epochs)
logging.info('Finish init model_sim. val_loss: {:.4f}'.format(ms_valid_loss))


# In[5]:


# imitation learning
logging.info('Start imitation learning...')
explorer_sim.run_k_episodes(il_episodes, 'train', update_memory=True, imitation_learning=True)
trainer.set_learning_rate(il_learning_rate)
trainer.optimize_epoch(il_epochs)
torch.save(model.state_dict(), il_weight_file)
logging.info('Finish imitation learning. Weights saved.')
logging.info('Experience set size: %d/%d', len(memory), memory.capacity)
explorer_sim.update_target_model(model)


# In[6]:


# reinforcement learning
policy.set_env(env_sim)
robot.set_policy(policy)
robot.print_info()
trainer.set_learning_rate(rl_learning_rate)
episode = 0 

while episode < train_episodes:
    if args.resume:
        epsilon = epsilon_end
    else:
        if episode < epsilon_decay:
            epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
        else:
            epsilon = epsilon_end
    epsilon = epsilon_end # fix small epsilon
    robot.policy.set_epsilon(epsilon)

    # evaluate the model
    if episode % evaluation_interval == 0:
        logging.info("Val in real...")
        policy.set_env(env)
        explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        logging.info("Val in sim...")
        policy.set_env(env_sim)
        explorer_sim.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        
    # explore real to train sim
    if sim_only == False:
        policy.set_env(env)
        explorer.run_k_episodes(sample_episodes, 'train', update_memory=False, update_raw_ob=True)
        trainer_sim.optimize_epoch(model_sim_epochs)

    # explore sim to train policy
    policy.set_env(env_sim)
    explorer_sim.run_k_episodes(sample_episodes_in_sim, 'train', update_memory=True, episode=episode)
    trainer.optimize_batch(train_batches)
    episode += 1

    if episode % target_update_interval == 0:
        explorer_sim.update_target_model(model)

    if episode != 0 and episode % checkpoint_interval == 0:
        torch.save(model.state_dict(), rl_weight_file)


# In[ ]:


# final test
policy.set_env(env)
explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)


# In[ ]:




