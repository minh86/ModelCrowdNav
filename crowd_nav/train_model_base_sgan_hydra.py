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
import torch
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

sys.path.append('../')
from crowd_sim.envs.utils.robot import Robot
from crowd_nav.utils.trainer import Trainer
from crowd_nav.utils.trainer_sgan import *
from crowd_nav.utils.memory import ReplayMemory
from crowd_nav.utils.explorer import Explorer
from crowd_nav.utils.datagen import DataGen
from crowd_nav.policy.policy_factory import policy_factory
from crowd_nav.policy.world_model import *
from crowd_nav.utils.misc import *

@hydra.main(config_path="runs")
def my_app(cfg: DictConfig) -> None:
    args = cfg.args
    output_dir = os.getcwd()

    hydra_config_name = HydraConfig.get().job.config_name + ".yaml"
    hydra_config_path = HydraConfig.get().runtime.config_sources[1].path
    hydra_config_fullpath = os.path.join(hydra_config_path, hydra_config_name)

    # configure paths
    shutil.copy(args.env_config, output_dir)
    shutil.copy(args.policy_config, output_dir)
    shutil.copy(args.train_config, output_dir)
    policy_config_file = os.path.join(output_dir, os.path.basename(args.policy_config))
    env_config_file = os.path.join(output_dir, os.path.basename(args.env_config))
    train_config_file = os.path.join(output_dir, os.path.basename(args.train_config))

    # log_file = os.path.join(output_dir, 'output.log')
    il_weight_file = os.path.join(output_dir, 'il_model.pth')
    rl_weight_file = os.path.join(output_dir, 'rl_model.pth')
    last_rl_weight_file = os.path.join(output_dir, 'last_rl_model.pth')

    # configure logging
    mode = 'a' if args.resume else 'w'
    # file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO if not args.debug else logging.DEBUG
    # logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
    logging.basicConfig(level=level, handlers=[stdout_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
    device = torch.device(args.device)
    logging.info('Using device: %s', device)
    mem_path = os.path.join(output_dir, 'memory.data')
    rawob_path = os.path.join(output_dir, 'rawob.data')
    model_sim_checkpoint = os.path.join(output_dir, 'model_sim.pt')

    # configure policy
    policy = policy_factory[args.policy]()
    if not policy.trainable:
        raise ValueError('Policy has to be trainable')
    if args.policy_config is None:
        raise ValueError('Policy config has to be specified for a trainable network')
    policy_config = configparser.RawConfigParser()
    policy_config.read(policy_config_file)
    policy.configure(policy_config)
    policy.set_device(args.device)
    policy.kinematics = args.kinematics

    # configure environment
    env_config = configparser.RawConfigParser()
    env_config.read(env_config_file)
    env = gym.make('CrowdSim-v0')
    env.configure(env_config)
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)
    logging.info("Val size: %d", env.case_size['val'])
    logging.info("Test size: %d", env.case_size['test'])

    train_config = configparser.RawConfigParser()
    train_config.read(train_config_file)
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
    train_render_interval = train_config.getint('train', 'train_render_interval')

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
    view_distance = train_config.getint('train_sim', 'view_distance')
    view_human = train_config.getint('train_sim', 'view_human')

    # dataset config
    train_datapath = train_config.get('dataset', 'train_datapath')
    val_datapath = train_config.get('dataset', 'val_datapath')
    test_datapath = train_config.get('dataset', 'test_datapath')
    stride = train_config.getint('dataset', 'stride')
    windows_size = train_config.getint('dataset', 'windows_size')

    max_human = args.human_num
    env.human_num = max_human
    if args.gradual:
        seq_success = ReplayMemory(num_epi_in_count)
        max_human = 1

    # ----------------------------  neptune params --------------------------------
    params = {"output_dir": output_dir, "model_sim_lr": model_sim_lr, "train_world_epochs": train_world_epochs,
              "sample_episodes_in_real_before_train": sample_episodes_in_real_before_train,
              "sample_episodes_in_sim": sample_episodes_in_sim, "init_train_episodes": init_train_episodes,
              "train_episodes": train_episodes, "target_update_interval": target_update_interval,
              "device": args.device, "world_model": args.world_model, "epsilon_start": epsilon_start,
              "epsilon_end": epsilon_end, "epsilon_decay": epsilon_decay, "num_epi_in_count": num_epi_in_count,
              "target_average_success": target_average_success, "train_dataset": os.path.basename(train_datapath),
              "val_dataset": os.path.basename(val_datapath), "test_dataset": os.path.basename(test_datapath),
              "view_distance": view_distance}

    # configure trainer and explorer
    memory = ReplayMemory(capacity)
    model = policy.get_model()
    batch_size = train_config.getint('trainer', 'batch_size')
    trainer = Trainer(model, memory, device, batch_size)
    explorer = Explorer(env, robot, device, memory, policy.gamma, target_policy=policy)
    explorer.rawob = ReplayMemory(capacity)
    explorer.raw_memory = ReplayMemory(capacity)

    # ----------------   config SGAN   ------------------
    os.makedirs(os.path.join(output_dir, 'cacheFiles/'))
    os.makedirs(os.path.join(output_dir, 'cacheFiles/train'))
    os.makedirs(os.path.join(output_dir, 'cacheFiles/val'))
    os.makedirs(os.path.join(output_dir, 'cacheFiles/test'))
    cacheFile_val = os.path.join(output_dir, 'cacheFiles/val/')
    cacheFile_train = os.path.join(output_dir, 'cacheFiles/train/')
    cacheFile_test = os.path.join(output_dir, 'cacheFiles/test/')
    cacheFile_generate = os.path.join(output_dir, 'cacheFiles/generate.txt')

    env_sim = gym.make('ModelCrowdSim-v0')
    env_sim.configure(env_config)
    if not policy.multiagent_training:
        env_sim.human_num = 1
    env_sim.set_robot(robot)
    env_sim.device = device
    model_sim = SGANWorld(cacheFile_generate, device,
                          obs_len=args.obs_len, time_step=env_sim.time_step,
                          pretrainPath=args.pretrainPath)  # SGAN model
    env_sim.sim_world = model_sim
    env.device = device
    env.sim_world = model_sim
    env_sim.add_noise = args.add_noise
    env_sim.use_linear_to_gen = args.use_linear_to_gen

    # model based things
    explorer_sim = Explorer(env_sim, robot, device, memory, policy.gamma, target_policy=policy)
    robot.set_policy(policy)

    # reinforcement learning
    policy.set_env(env)
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
        run["config/env"].upload(env_config_file)
        run["config/policy"].upload(policy_config_file)
        run["config/train"].upload(train_config_file)
        run["config/hydra"].upload(hydra_config_fullpath)

    # ============  Using dataset  ===============

    if not args.use_dataset:
        logging.info("Collect %s trajectories from simulation..." % (sample_episodes_in_real_before_train))
        explorer.run_k_episodes(sample_episodes_in_real_before_train, 'train', update_memory=False, update_raw_ob=True,
                                stay=True)
    else:  # -----------  Using trajnet++ dataset  ------------
        train_sl = None
        test_sl = None
        Store_for_world_fn = StoreAction  # Update this for SGAN
        if args.cutting_point > 0:
            train_sl = [0, args.cutting_point]
            test_sl = [args.cutting_point, env.case_size['test']]
        logging.info("Collect data from dataset (trajnet++)...")
        # load data for training value network (padding moving)
        phase = "train"
        if args.no_val:
            phase = "all"
        else:
            # load data for validation and testing
            val_raw_memory, _ = GetRealData(dataset_file=val_datapath, phase="val",
                                            stride=stride, windows_size=windows_size, dataset_slice=train_sl,
                                            Store_for_world_fn=Store_for_world_fn, cacheFile=cacheFile_val)
        # load data for training world model (padding stay)
        # _, rawob = GetRealData(dataset_file=train_datapath, phase=phase, stride=stride,
        #                        windows_size=windows_size,slide=train_sl)
        train_raw_memory, rawob = GetRealData(dataset_file=train_datapath, phase=phase, stride=stride,
                                              windows_size=windows_size, dataset_slice=train_sl,
                                              Store_for_world_fn=Store_for_world_fn, cacheFile=cacheFile_train)
        test_raw_memory, _ = GetRealData(dataset_file=test_datapath, phase="test", stride=stride,
                                         windows_size=windows_size,
                                         dataset_slice=test_sl, Store_for_world_fn=Store_for_world_fn,
                                         cacheFile=cacheFile_test)
        data_generator.raw_memory = train_raw_memory

    # ============  training world model  ===============
    if args.pretrainPath == "":
        trainer_sgan = Trainer_SGAN(output_dir, obs_len=args.obs_len, device=device)
        logging.info("Training world model...")
        ms_valid_loss, generator = trainer_sgan.run_train()
        model_sim.generator = generator
        logging.info('Model-based env. Total loss: {:.4f}'.format(ms_valid_loss))

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
    _, success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(il_episodes,
                                                                                                max_human=max_human,
                                                                                                imitation_learning=True,
                                                                                                # random_robot=False,
                                                                                                add_sim=(
                                                                                                    not args.real_only),
                                                                                                # random_epi=False,
                                                                                                # render_path=output_dir,
                                                                                                replace_robot=args.use_dataset,
                                                                                                # stay=True,
                                                                                                view_distance=view_distance,
                                                                                                view_human=view_human,
                                                                                                sgan_genfile=cacheFile_generate,
                                                                                                min_end=args.obs_len,
                                                                                                static_end=args.obs_len,
                                                                                                )
    video_tag = "il_vi"
    explorer_sim.env.render("video", os.path.join(output_dir, video_tag + "_ep" + ".gif"))
    if args.neptune:
        Resize_GIF(os.path.join(output_dir, video_tag + "_ep" + ".gif"))
        run[video_tag + "/" + video_tag + "_ep" + ".gif"].upload(
            os.path.join(output_dir, video_tag + "_ep" + ".gif"))  # upload to neptune
    logging.info("Training on imitation experience...")
    trainer.optimize_epoch(il_epochs)
    data_generator.policy = policy
    robot.set_policy(policy)

    # ==============   gen data by explorer in mix reality  ================
    logging.info("Training phase...")
    best_cumulative_rewards = float('-inf')
    mem_success = ReplayMemory(num_epi_in_count, 0)
    mem_collision = ReplayMemory(num_epi_in_count, 0)
    mem_timeout = ReplayMemory(num_epi_in_count, 0)
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

        # --------------  train world model  --------------
        if args.retrain_world_model:
            if args.reinit_world:
                model_sim.apply(init_weight)  # reinit weight before training
            # ms_valid_loss = trainer_sgan.run_train()
            if (episode + 1) % args.sgan_train_per_episode == 0 and args.pretrainPath == "":
                logging.info("Training world model...")
                ms_valid_loss, generator = trainer_sgan.run_train()
                model_sim.generator = generator
                if args.neptune:
                    run["train_world_model/loss"].log(ms_valid_loss)  # log to neptune
                    if args.gradual:
                        run["Current human num"].log(max_human)  # log to neptune

        # let's explore mix reality!
        if args.use_dataset:
            data_generator.raw_memory = train_raw_memory
        _, success, collision, timeout = data_generator.gen_data_from_explore_in_mix(sample_episodes_in_sim,
                                                                                     add_sim=(not args.real_only),
                                                                                     max_human=max_human, phase='train',
                                                                                     view_distance=view_distance,
                                                                                     view_human=view_human,
                                                                                     replace_robot=args.use_dataset,
                                                                                     sgan_genfile=cacheFile_generate,
                                                                                     min_end=args.obs_len, )
        mem_success.push(success)
        mem_collision.push(collision)
        mem_timeout.push(timeout)
        total_epi = sum(mem_success.memory) + sum(mem_timeout.memory) + sum(mem_collision.memory)
        if args.neptune:
            run["exp_in_mix/success_rate"].log(sum(mem_success.memory) / total_epi)  # log to neptune
            run["exp_in_mix/collision_rate"].log(sum(mem_collision.memory) / total_epi)  # log to neptune
            run["exp_in_mix/timeout_rate"].log(sum(mem_timeout.memory) / total_epi)  # log to neptune

        if (episode + 1) % train_render_interval == 0 or episode == 0:
            video_tag = "train_vi"
            explorer_sim.env.render("video", os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            if args.neptune:
                Resize_GIF(os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
                run[video_tag + "/" + video_tag + "_ep" + str(episode) + ".gif"].upload(
                    os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune

        # train value network
        average_loss = trainer.optimize_batch(train_batches)
        if args.neptune:
            run["train_value_network/loss"].log(average_loss)  # log to neptune
            run["train_value_network/PositiveRate"].log(PositiveRate(memory))  # log to neptune
        # logging.info('Policy model env. val_loss: {:.4f}'.format(average_loss))

        # evaluate the model
        if (episode + 1) % evaluation_interval == 0 and not args.no_val:
            logging.info("Val in real...")
            video_tag = "val_vi"
            if args.use_dataset:
                data_generator.raw_memory = val_raw_memory
                data_generator.counter = 0
                cumulative_rewards, success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(
                    data_generator.count(),  # val size for crowds_students001.ndjson
                    # max_human=max_human,
                    random_robot=False,
                    add_sim=False,
                    random_epi=False,
                    phase='val',
                    # render_path=output_dir,
                    view_distance=view_distance,
                    view_human=view_human,
                    returnRate=True,
                    replace_robot=args.use_dataset,
                    sgan_genfile=cacheFile_generate,
                    min_end=args.obs_len,
                )
                explorer_sim.env.render("video",
                                        os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            else:
                policy.set_env(env)
                cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(
                    env.case_size['val'],
                    'val', episode=episode)
                explorer.env.render("video", os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
            if args.neptune:
                run["val/success_rate"].log(success_rate)  # log to neptune
                run["val/collision_rate"].log(collision_rate)  # log to neptune
                run["val/timeout_rate"].log(timeout_rate)  # log to neptune
                Resize_GIF(os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
                run[video_tag + "/" + video_tag + "_ep" + str(episode) + ".gif"].upload(
                    os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune

            if cumulative_rewards > best_cumulative_rewards and args.no_val == False:
                best_cumulative_rewards = cumulative_rewards
                torch.save(model.state_dict(), rl_weight_file)
                logging.info("Best RL model saved!")
                if args.neptune:
                    run["model/best_val"].upload(rl_weight_file)

        # update target model
        if (episode + 1) % target_update_interval == 0:
            data_generator.update_target_model(model)
            torch.save(model.state_dict(), last_rl_weight_file)

    # final test
    logging.info("Testing by %d episodes...", env.case_size['test'])
    video_tag = "test_vi"
    policy.set_env(env)
    torch.save(model.state_dict(), last_rl_weight_file)
    if args.neptune:
        run["model/best_val"].upload(last_rl_weight_file)
    if not args.no_val:  # load model from validation
        logging.info("Load best RL model")
        robot.policy.model.load_state_dict(torch.load(rl_weight_file))  # load best model

    if args.use_dataset:
        data_generator.raw_memory = test_raw_memory
        data_generator.counter = 0
        cumulative_rewards, success_rate, collision_rate, timeout_rate = data_generator.gen_data_from_explore_in_mix(
            data_generator.count(),  # test size for crowds_students003.ndjson
            # max_human=max_human,
            random_robot=False,
            add_sim=False,
            random_epi=False,
            phase='test',
            # render_path=output_dir,
            view_distance=view_distance,
            view_human=view_human,
            returnRate=True,
            replace_robot=args.use_dataset,
            sgan_genfile=cacheFile_generate,
            min_end=args.obs_len,
        )
        explorer_sim.env.render("video", os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
    else:
        cumulative_rewards, success_rate, collision_rate, timeout_rate = explorer.run_k_episodes(env.case_size['test'],
                                                                                                 'test',
                                                                                                 episode=episode)
        explorer.env.render("video", os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))

    if args.neptune:
        run["test/success_rate"].log(success_rate)  # log to neptune
        run["test/collision_rate"].log(collision_rate)  # log to neptune
        run["test/timeout_rate"].log(timeout_rate)  # log to neptune
        Resize_GIF(os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))
        run[video_tag + "/" + video_tag + "_ep" + str(episode) + ".gif"].upload(
            os.path.join(output_dir, video_tag + "_ep" + str(episode) + ".gif"))  # upload to neptune
        run.stop()

if __name__ == '__main__':
    my_app()