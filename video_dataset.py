import torch
from torch.utils.data import Dataset

import torchvision
import av

import copy

import youtube_dl
import yt_dlp
import subprocess
import datetime

import numpy as np

import pandas as pd
import simplejson as json
import tqdm.notebook as tqdm

import os

from text_transform import LabelTextEncode
from image_transforms import get_transform

class VideoDataset(Dataset):
    def __init__(self, dataset, data_dir, annot_files, args, is_valid):
        self.data = []
        self.data_dir = data_dir
        self.label_transform = LabelTextEncode(dataset)
        self.actions = set()
        self.transform = get_transform(is_valid)
        self.args = args
        limit = args.limit
        for annot_file in annot_files:
            with open(os.path.join(data_dir, annot_file)) as f:
                data = json.load(f)
                if dataset == 'activity-net':
                    data = data['database']
            for k in data.keys():
                data[k]['key'] = k
            current = []
            for k in data.keys():
                d = data[k]
                if dataset == 'kinetics':
                    d['annotations'] = [d['annotations']]
                for annot in d['annotations']:
                    new_action = copy.deepcopy(d)
                    new_action['annotations'] = annot
                    current.append(new_action)
            self.data.extend(current)
        if limit != -1:
            self.data = self.data[:limit]
        remove = set()
        for d in tqdm.tqdm(self.data):
            self.actions.add(d['annotations']['label'])
            time = [str(datetime.timedelta(seconds=int(t))) for t in d['annotations']['segment']]
            target = os.path.join(data_dir, d['key'] + str(time[0]) + '_' + str(time[1]) + '.mkv')
            d['file'] = target
            print(target, os.path.exists(target))
            try:
                if not os.path.exists(target):
                    subprocess.check_call('ffmpeg -ss "%s" -to "%s" -i "$(yt-dlp -f best --get-url "%s")" -c:v copy  "%s"temp.mkv; ffmpeg -i "%s"temp.mkv -filter:v scale=128:72 -c:a copy "%s"'                
                        % (*time, d['url'], target[:-4], target[:-4], target), shell=True)
            except:
                remove.add(d['file'])
        print(remove)
        self.data = [d for d in self.data if d['file'] not in remove]
        self.actions = [self.label_transform(action) for action in self.actions]
        self.action2idx = {action:i for i, action in enumerate(self.actions)}
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        text = d['annotations']['label']
        text = self.action2idx[self.label_transform(text)]
        time = torchvision.io.read_video_timestamps(d['file'])
        length = len(time[0])

        start = np.random.randint(max(length - self.args.clip_len, 1))
        end = min(start + self.args.clip_len - 1, length)
        video = torchvision.io.read_video(d['file'], start_pts = time[0][start], end_pts = time[0][end])
        video = video[0]
        video = video[:self.args.clip_len]
        video = self.transform(video)
        temp = torch.zeros(self.args.clip_len, *video.shape[1:])
        temp[:video.size(0)] = video
        video = temp

        
        return (video, text)
