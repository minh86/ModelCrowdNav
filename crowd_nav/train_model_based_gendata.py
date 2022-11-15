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
parser.add_argument('--add_positive', default=False, action='store_true') # adding fake positive experience to combat timeout
parser.add_argument('--gradual', default=False, action='store_true') # gradually changing human num
parser.add_argument('--reinit_world', default=False, action='store_true') # gradually changing human num
parser.add_argument('--human_num', type=int, default=5)


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
last_rl_weight_file = os.path.join(args.output_dir, 'last_rl_model.pth')

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
num_epi_in_count = train_config.getint('train_sim', 'num_epi_in_count')
target_average_success = train_config.getfloat('train_sim', 'target_average_success')

max_human = args.human_num
env.human_num = max_human
if args.gradual:
    seq_success = ReplayMemory(num_epi_in_count)
    max_human = 1
# ----------------------------  neptune params --------------------------------
params ={"output_dir":args.output_dir, "model_sim_lr":model_sim_lr, "train_world_epochs": train_world_epochs,
         "sample_episodes_in_real_before_train": sample_episodes_in_real_before_train,
         "sample_episodes_in_sim": sample_episodes_in_sim, "init_train_episodes":init_train_episodes,
         "train_episodes":train_episodes,"target_update_interval":target_update_interval,
         "device": args.device, "world_model": args.world_model, "epsilon_start": epsilon_start,
         "epsilon_end":epsilon_end, "epsilon_decay":epsilon_decay, "num_epi_in_count":num_epi_in_count,
         "target_average_success": target_average_success}

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
    run["config/env"].upload(args.env_config)
    run["config/policy"].upload(args.policy_config)
    run["config/train"].upload(args.train_config)

# ============  training world model  ===============
# explore real to train sim
logging.info("Collect data and init phase...")
explorer.run_k_episodes(sample_episodes_in_real_before_train, 'train', update_memory=False, update_raw_ob=True, stay=True)
logging.info("Training world model...")
ms_valid_loss = trainer_sim.optimize_epoch(train_world_epochs)
logging.info('Model-based env.  val_loss: {:.4f}'.format(ms_valid_loss))

# ============ imitation learning  ====================
logging.info("Start imitation learning...")
il_episodes = train_config.getint('imitation_learning', 'il_episodes')
il_policy = train_config.get('imitation_learning', 'il_policy')
il_epochs = train_config.getint('imitation_learning', 'il_epochs')
il_learning_rate = train_config.getfloat('imitation_learning', 'il_learning_rate')
trainer.set_learning_rate(il_learning_rate)
if robot.visible:
    safety_space = 0
else:
    safety_space = train_config.getfloat('imitation_learning', 'safety_space')
il_policy = policy_factory[il_policy]()
il_policy.multiagent_training = policy.multiagent_training
il_policy.safety_space = safety_space
il_policy.time_step = policy.time_step
il_policy.device = policy.device
data_generator.policy = il_policy
robot.set_policy(il_policy)
success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(il_episodes,
                                                                                         max_human=max_human,
                                                                                         imitation_learning=True)
video_tag = "il_vi"
explorer_sim.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + ".gif"))
if args.neptune:
    Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + ".gif"))
    run[video_tag + "/" + video_tag + "_ep" + ".gif"].upload(
        os.path.join(args.output_dir, video_tag + "_ep" + ".gif"))  # upload to neptune

trainer.optimize_epoch(il_epochs)
data_generator.policy = policy
robot.set_policy(policy)

# # =======================  init by imagination  ==================
# for episode in tqdm(range(init_train_episodes)):
#     # # gen sim data and train
#     data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=True, imitation_learning=True)
#     data_generator.gen_new_data(sample_episodes_in_sim, reach_goal=False, imitation_learning=True)
#
#     # # gen sim trajectories data from real data
#     # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=True)
#     # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=False)
#
#     # gen sim trajectories data from mix sim-real data
#     # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=True, add_sim=True,max_human=max_human, imitation_learning=True)
#     # data_generator.gen_new_data_from_real(sample_episodes_in_sim, reach_goal=False, add_sim=True, max_human=max_human, imitation_learning=True)
#     memory.shuffle()
#     average_loss = trainer.optimize_batch(train_batches)
#     if args.neptune:
#         run["train_value_network_init/loss"].log(average_loss) # log to neptune
#
#     # update target model
#     if (episode+1) % target_update_interval == 0:
#         data_generator.update_target_model(model)

# ==============   gen data by explorer in mix reality  ================
logging.info("Training phase...")
best_cumulative_rewards = float('-inf')
mem_success = ReplayMemory(num_epi_in_count)
mem_collision = ReplayMemory(num_epi_in_count)
mem_timeout = ReplayMemory(num_epi_in_count)
# logging.info("Load the best RL model")
# robot.policy.model.load_state_dict(torch.load(rl_weight_file))  # load best model
data_generator.update_target_model(robot.policy.model)
for episode in tqdm(range(train_episodes)):
    if args.neptune:
        run["Current Episode"].log("%d / %d" % (episode, train_episodes))  # log to neptune

    if episode < epsilon_decay:
        epsilon = epsilon_start + (epsilon_end - epsilon_start) / epsilon_decay * episode
    else:
        epsilon = epsilon_end
    robot.policy.set_epsilon(epsilon)

    # train world model
    if args.reinit_world:
        model_sim.apply(init_weight)  # reinit weight before training
    ms_valid_loss = trainer_sim.optimize_epoch(train_world_epochs, reset=args.reinit_world)
    if args.neptune:
        run["train_world_model/loss"].log(ms_valid_loss)  # log to neptune
        if args.gradual:
            run["Current human num"].log(max_human)  # log to neptune

    # gradually changing difficult env level
    if args.gradual:
        if sum(seq_success.memory) >= target_average_success * num_epi_in_count and max_human < env.human_num:
            max_human += 1
            seq_success.clear()

    # let's explore mix reality!
    success, collision, timeout = data_generator.gen_data_from_explore_in_mix(sample_episodes_in_sim, max_human=max_human)
    mem_success.push(success); mem_collision.push(collision); mem_timeout.push(timeout)
    total_epi = sum(mem_success.memory) + sum(mem_timeout.memory) + sum(mem_collision.memory)
    if args.neptune:
        run["exp_in_mix/success_rate"].log(sum(mem_success.memory)/total_epi)  # log to neptune
        run["exp_in_mix/collision_rate"].log(sum(mem_collision.memory)/total_epi)  # log to neptune
        run["exp_in_mix/timeout_rate"].log(sum(mem_timeout.memory)/total_epi)  # log to neptune

    if args.gradual:
        seq_success.push(success_rate)

    if (episode + 1) % train_render_interval == 0 or episode==0:
        video_tag = "train_vi"
        explorer_sim.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
        if args.neptune:
            Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            run[video_tag+"/"+video_tag + "_ep" + str(episode) + ".gif"].upload(
                os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune

    # train value network
    average_loss = trainer.optimize_batch(train_batches)
    if args.neptune:
        run["train_value_network/loss"].log(average_loss)  # log to neptune
        run["train_value_network/PositiveRate"].log(PositiveRate(memory))  # log to neptune
    # logging.info('Policy model env. val_loss: {:.4f}'.format(average_loss))

    # evaluate the model
    if (episode + 1) % evaluation_interval == 0 and not args.no_val:
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
            if args.neptune:
                run["model/best_val"].upload(rl_weight_file)

    # update target model
    if (episode+1) % target_update_interval == 0:
        data_generator.update_target_model(model)

# final test
logging.info("Testing by %d episodes...", env.case_size['test'])
policy.set_env(env)
torch.save(model.state_dict(), last_rl_weight_file)
if args.neptune:
    run["model/best_val"].upload(last_rl_weight_file)
if not args.no_val: # load model from validation
    logging.info("Load best RL model")
    robot.policy.model.load_state_dict(torch.load(rl_weight_file))  # load best model
cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(env.case_size['test'], 'test', episode=episode)
video_tag="test_vi"
explorer.env.render("video", os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
if args.neptune:
    run["test/success_rate"].log(success_rate)  # log to neptune
    run["test/collision_rate"].log(collision_rate)  # log to neptune
    run["test/timeout_rate"].log(timeout_rate)  # log to neptune
    Resize_GIF(os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))
    run[video_tag+"/"+video_tag + "_ep" + str(episode) + ".gif"].upload(
                os.path.join(args.output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune
    run.stop()
