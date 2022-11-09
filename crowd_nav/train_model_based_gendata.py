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
import neptune.new as neptune
import numpy as np

sys.path.append('../')
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.trainer_sim import Trainer_Sim
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.datagen import DataGen
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.world_model import *
from crowd_nav.utils.misc import *

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
parser.add_argument('--add_noise', default=False, action='store_true')
parser.add_argument('--dyna', default=False, action='store_true')
parser.add_argument('--no_val', default=False, action='store_true')
parser.add_argument('--use_linear_to_gen', default=False, action='store_true')
parser.add_argument('--neptune', default=False, action='store_true')
parser.add_argument('--world_model', type=str, default='mlp')
parser.add_argument('--neptune_name', type=str, default='Untitled')
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
logging.info("Val size: %d",env.case_size['val'])
logging.info("Test size: %d",env.case_size['test'])

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
train_render_interval =  train_config.getint('train', 'train_render_interval')

model_sim_lr = train_config.getfloat('train_sim', 'model_sim_lr')
train_world_epochs = train_config.getint('train_sim', 'train_world_epochs')
sample_episodes_in_real = train_config.getint('train_sim', 'sample_episodes_in_real')
sample_episodes_in_real_before_train = train_config.getint('train_sim', 'sample_episodes_in_real_before_train')
ms_batchsize = train_config.getint('train_sim', 'ms_batchsize')
sample_episodes_in_sim = train_config.getint('train_sim', 'sample_episodes_in_sim')
init_train_episodes = train_config.getint('train_sim', 'init_train_episodes')
api_token = train_config.get('neptune', 'api_token')
neptune_project = train_config.get('neptune', 'neptune_project')

# ----------------------------  neptune params --------------------------------
params ={"output_dir":args.output_dir, "model_sim_lr":model_sim_lr, "train_world_epochs": train_world_epochs,
         "sample_episodes_in_real_before_train": sample_episodes_in_real_before_train,
         "sample_episodes_in_sim": sample_episodes_in_sim, "init_train_episodes":init_train_episodes,
         "train_episodes":train_episodes,"target_update_interval":target_update_interval,
         "device": args.device, "world_model": args.world_model, "epsilon_start": epsilon_start,
         "epsilon_end":epsilon_end, "epsilon_decay":epsilon_decay}

# configure trainer and explorer
memory = ReplayMemory(capacity)
model = policy.get_model()
batch_size = train_config.getint('trainer', 'batch_size')
trainer = Trainer(model, memory, device, batch_size)
explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)
explorer.rawob = ReplayMemory(capacity)
explorer.raw_memory = ReplayMemory(capacity)

# config sim environment
if args.world_model =="mlp":
    model_sim = MlpWorld(env_config.getint('sim', 'human_num'),multihuman=policy.multiagent_training)
if args.world_model =="attention":
    model_sim = AttentionWorld()
model_sim.to(device)
model_sim.device = device
env_sim = gym.make('ModelCrowdSim-v0')
env_sim.configure(env_config)
if not policy.multiagent_training:
    env_sim.human_num = 1
env_sim.set_robot(robot)
env_sim.device = device
env_sim.sim_world = model_sim
env.device = device
env.sim_world = model_sim
env_sim.add_noise = args.add_noise
env_sim.use_linear_to_gen = args.use_linear_to_gen

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
data_generator = DataGen(memory, robot, env_sim, policy)
data_generator.update_target_model(model)
data_generator.raw_memory = explorer.raw_memory
episode = 0

# ============== neptune things  ================
if args.neptune:
    run = neptune.init_run(
        project=neptune_project,
        api_token=api_token,
        name=args.neptune_name
    )
    run["parameters"] = params

# ============  init and collect data  ===============
# explore real to train sim
logging.info("Collect data and init phase...")
explorer.run_k_episodes(sample_episodes_in_real_before_train, 'train', update_memory=False, update_raw_ob=True, stay=True)
logging.info("Training world model...")
ms_valid_loss = trainer_sim.optimize_epoch(train_world_epochs)
logging.info('Model-based env.  val_loss: {:.4f}'.format(ms_valid_loss))

best_cumulative_rewards = float('-inf')
update_real_memory = False
for episode in tqdm(range(init_train_episodes)):
    # # explore in real
    # if args.dyna:
    #     update_real_memory = True
    # policy.set_env(env)
    # explorer.run_k_episodes(sample_episodes_in_real, 'train', update_memory=update_real_memory, update_raw_ob=True, stay=True)
    # ms_valid_loss = trainer_sim.optimize_epoch(train_world_epochs)

    # gen sim data and train
    data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=True, imitation_learning=True)
    data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=False, imitation_learning=True)

    # # gen sim trajectories data from real data
    # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=True)
    # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=False)
    # # gen sim trajectories data from mix sim-real data
    # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=True, add_sim=True)
    # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=False, add_sim=True)

    average_loss = trainer.optimize_batch(train_batches)
    if args.neptune:
        run["train_value_network_init/loss"].log(average_loss) # log to neptune
    # logging.info('Policy model env. val_loss: {:.4f}'.format(average_loss))

    # # evaluate the model
    # if (episode+1) % evaluation_interval == 0 and episode != 0 :
    #     logging.info("Val in real...")
    #     policy.set_env(env)
    #     video_tag = "im_val"
    #     cumulative_rewards = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
    #     explorer.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
    #     if cumulative_rewards > best_cumulative_rewards and args.no_val == False:
    #         best_cumulative_rewards = cumulative_rewards
    #         torch.save(model.state_dict(), rl_weight_file)
    #         logging.info("Best RL model saved!")

    # update target model
    if (episode+1) % target_update_interval == 0:
        data_generator.update_target_model(model)

# ==============   gen data by explorer in mix reality  ================
logging.info("Training phase...")
best_cumulative_rewards = float('-inf')
# logging.info("Load the best RL model")
# robot.policy.model.load_state_dict(torch.load(rl_weight_file))  # load best model
data_generator.update_target_model(robot.policy.model)
for episode in tqdm(range(train_episodes)):
    if episode < epsilon_decay:
        epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
    else:
        epsilon = epsilon_end
    robot.policy.set_epsilon(epsilon)
    # retrain world model before gen data
    # model_sim.apply(init_weight)  # reinit weight before training
    ms_valid_loss = trainer_sim.optimize_epoch(train_world_epochs)
    if args.neptune:
        run["train_world_model/loss"].log(ms_valid_loss)  # log to neptune

    # adding positive fake experience to battle timeout
    probability = np.random.random()
    if probability < epsilon-epsilon_end:
        data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=True, add_sim=True, imitation_learning=True)

    # let's explore mix reality!
    success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(sample_episodes_in_sim)
    if args.neptune:
        run["exp_in_mix/success_rate"].log(success_rate)  # log to neptune
        run["exp_in_mix/collision_rate"].log(collision_rate)  # log to neptune
        run["exp_in_mix/timeout_rate"].log(timeout_rate)  # log to neptune

    if (episode + 1) % train_render_interval == 0 and episode != 1:
        video_tag = "train_vi"
        explorer_sim.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
        if args.neptune:
            Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            run[video_tag+"/"+video_tag + "_ep" + str(episode) + ".gif"].upload(
                os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune

    average_loss = trainer.optimize_batch(train_batches)
    if args.neptune:
        run["train_value_network/loss"].log(average_loss)  # log to neptune
    # logging.info('Policy model env. val_loss: {:.4f}'.format(average_loss))

    # evaluate the model
    if (episode + 1) % evaluation_interval == 0 and episode != 1 and not args.no_val:
        logging.info("Val in real...")
        policy.set_env(env)
        cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(env.case_size['val'], 'val', episode=episode)
        if args.neptune:
            run["val/success_rate"].log(success_rate)  # log to neptune
            run["val/collision_rate"].log(collision_rate)  # log to neptune
            run["val/timeout_rate"].log(timeout_rate)  # log to neptune
        video_tag = "val_vi"
        explorer.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
        if args.neptune:
            Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            run[video_tag+"/"+video_tag + "_ep" + str(episode) + ".gif"].upload(
                os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif")) # upload to neptune

        if cumulative_rewards > best_cumulative_rewards and args.no_val == False:
            best_cumulative_rewards = cumulative_rewards
            torch.save(model.state_dict(), rl_weight_file)
            logging.info("Best RL model saved!")

    # update target model
    if (episode+1) % target_update_interval == 0:
        data_generator.update_target_model(model)

# final test
logging.info("Testing by %d episodes...", env.case_size['test'])
policy.set_env(env)
if not args.no_val: # load model from validation
    logging.info("Load best RL model")
    robot.policy.model.load_state_dict(torch.load(rl_weight_file))  # load best model
explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)
video_tag="test_vi"
explorer.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
if args.neptune:
    Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
    run[video_tag+"/"+video_tag + "_ep" + str(episode) + ".gif"].upload(
                os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune
    run.stop()
