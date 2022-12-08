import logging
import os.path

import numpy as np
from PIL import Image, ImageSequence
from trajnetplusplustools.reader import Reader
from trajnetplusplustools.data import SceneRow, TrackRow
from trajnetplusplustools.summarize import dataset_plots

from crowd_sim.envs.utils.state import ObservableState
from crowd_nav.utils.memory import ReplayMemory
from crowd_sim.envs.utils.info import *
import torch
from crowd_sim.envs.utils.action import ActionRot, ActionXY


def Resize_GIF(src_path):
    # Output (max) size
    size = 800, 600
    # Open source
    im = Image.open(src_path)
    # Get sequence iterator
    frames = ImageSequence.Iterator(im)

    # Wrap on-the-fly thumbnail generator
    def thumbnails(frames):
        for frame in frames:
            thumbnail = frame.copy()
            thumbnail.thumbnail(size, Image.ANTIALIAS)
            yield thumbnail

    frames = thumbnails(frames)
    # Save output
    om = next(frames)  # Handle first frame separately
    om.info = im.info  # Copy sequence info
    om.save(src_path, save_all=True, append_images=list(frames))


def PositiveRate(memory):
    pos = 0
    for _, value in memory.memory:
        if value.item() > 0:
            pos += 1
    return pos / len(memory.memory)


def GetRealData(dataset_file, phase="train", capacity=10000, stride=-1, windows_size=-1, padding_last="stay",
                padding_first="none", dataset_slice=None):
    # dataset_plots(dataset_file)
    reader = Reader(dataset_file, scene_type='both')
    reader.joinScene(stride, windows_size)  # Join multiple scenes into one
    limit = -1; start = 0; total = len(reader.scenes_by_id)
    if dataset_slice is not None:
        start = dataset_slice[0]
        total = dataset_slice[1]
        if phase == "test":
            limit = total
    if phase == "train":
        limit = int(0.7 * total)
    if phase == "val":
        start = int(0.7 * total)
        limit = total - start

    scenes = reader.scenes(limit=limit, start=start)
    raw_memory = ReplayMemory(capacity)
    rawob = ReplayMemory(capacity)
    count=0
    for scene_id, fps, pri_human, rows, paths in scenes:
        count+=1
        start = reader.scenes_by_id[scene_id].start
        end = reader.scenes_by_id[scene_id].end
        frames = range(start, end + 1)
        frame_ids = [r.frame
                     for frame in frames
                     for r in reader.tracks_by_frame.get(frame, [])]
        frame_ids = sorted([*{*frame_ids}])
        obs = Convert_to_ObserState(fps, paths, frame_ids, padding_last=padding_last, padding_first=padding_first)
        start_ends = [[p[0].x, p[0].y, p[-1].x, p[-1].y] for p in paths] # possible scenarios for robot
        for i, ob in enumerate(obs):
            done = False
            if i == len(obs) - 1:
                done = True
            raw_memory.push((ob, 0, done, Nothing(), start_ends))  # memory for datagen

            # memory for trainning world model
            if i > 0:
                current_s = [tmpo.getvalue() for tmpo in obs[i - 1]]
                next_s = [tmpo.getvalue() for tmpo in obs[i]]
                next_action = [s[2:] for s in next_s]
                next_action = next_action[:len(current_s)]
                rawob.push((torch.Tensor(current_s), torch.Tensor(next_action)))

    logging.info("Loaded %s cases in %s for phase: %s " % (count, os.path.basename(dataset_file), phase))
    return raw_memory, rawob


def Convert_to_ObserState(fps, paths, frame_ids, radius=0.3, padding_last="stay", padding_first="none"):
    obs = []
    for c_frame in frame_ids:
        obs.append(GetState(paths, c_frame, frame_ids, radius=radius, fps=fps, padding_last=padding_last, padding_first=padding_first))
    # tmp =[ob[5] if 5< len(ob) else [] for ob in obs]
    return obs


def GetState(paths, c_frame, frame_ids, radius=0.3, fps=2.5, padding_last="stay", padding_first="none"):
    state = []
    for p in paths:
        p_i = GetIndex(c_frame, p, frame_ids)
        if p_i == -1: # padding first
            if padding_first == "none":
                continue
            if padding_first == "stay":
                p_i = 0
        vx, vy = GetVel(p_i, p, fps)
        if p_i >= len(p):
            if padding_last == "stay":
                px, py = p[-1].x, p[-1].y # disappeared human is padding by its last position
            if padding_last == "moving":
                # disappeared human is padding by keep moving with last velocity
                last_vx, last_vy = GetVel(len(p)-1, p, fps)
                px = p[-1].x + (last_vx/fps) * (p_i-len(p))
                py = p[-1].y + (last_vy/fps) * (p_i-len(p))
                vx, vy = last_vx, last_vy
        else:
            px, py = p[p_i].x, p[p_i].y

        state.append(ObservableState(px, py, vx, vy, radius))
    return state

def GetIndex(frameid, path, frame_ids):
    if frameid < path[0].frame:
        return -1

    # return fake next id of the path
    if frameid > path[-1].frame:
        pass_f = [f for f in frame_ids if path[-1].frame < f <= frameid]
        return len(path) + len(pass_f)

    for i, frame in enumerate(path):
        if frame.frame == frameid:
            return i
    return -1


def GetVel(id, path, fps):
    if id == 0 or id >= len(path):
        return 0, 0
    last_id = id - 1
    vx = (path[id].x - path[last_id].x) * fps
    vy = (path[id].y - path[last_id].y) * fps
    return vx, vy
