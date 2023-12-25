
import csv
from torch.utils.data import Dataset
import pandas as pd
import os

from urllib.request import urlopen
from PIL import Image
import decord
from decord import cpu, gpu
import numpy as np
import torchvision.transforms as T
import ast
class MyDataset(Dataset):
    def __init__(self, data_path="", query_file="", image_transforms=None, fps=30, start_sample=0, max_samples=None):
        """Args:- path: path to the dataset
            - image_transforms: image transforms to apply to the frames
            - fps: fps of the video
            - query_file: path to the query file
            - max_samples: maximum number of samples to use"""
        self.path = data_path
        self.image_transforms = image_transforms
        self.fps = fps
        self.query_file = query_file
        self.start_sample = 0
        self.max_samples = max_samples

        with open(self.query_file) as f:
            self.df = pd.read_csv(f, index_col=None, keep_default_na=False)

        if self.max_samples is not None:
            self.df = self.df[start_sample : start_sample + self.max_samples]
        
        self.length = len(self.df)
    
    def get_sample_path(self, index):
        return os.path.join(self.path, self.df.iloc[index]["video_name"])

    def get_video(self, video_path, fps=30, sample_freq=None, transform=None):
        """Gets a video and returns it as a tensor.
            - video_path: path to the video
            - fps: fps of the video
            - sample_freq: frequency of sampling the video frames. sample_freq=10 means every 10th frame is sampled. if None, all frames are sampled
        """
         # If fixed width and height are required, VideoReader takes width and height as arguments.
        video_reader = decord.VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
        decord.bridge.set_bridge('torch')
        vlen = len(video_reader)
        #print(vlen)
        original_fps = video_reader.get_avg_fps()
        #print(original_fps)
        num_frames = int(vlen * fps / original_fps)
        # num_frames = min(self.max_num_frames, num_frames)
        frame_idxs = np.linspace(0, vlen, num_frames, endpoint=False).astype(np.int_)
        if sample_freq:
            frame_idxs = frame_idxs[::sample_freq]
        video = video_reader.get_batch(frame_idxs).byte()
        video = video.permute(0, 3, 1, 2)

        # Deafult transform is to transform .ToPILImage(). TODO: implement functionality for other transforms
        if not transform:
            transform = T.ToPILImage()
            video = [transform(video[i]) for i in range(0, video.shape[0])]

        return video
    
    def __getitem__(self, index):
        out_dict = self.df.iloc[index].to_dict()
        sample_path = self.get_sample_path(index)
        # Load and transform image
        video = self.get_video(sample_path)
        out_dict["video"] = video
        out_dict["index"] = index
        return out_dict

    def __len__(self):
        return self.length


"""
def get_data(filename):
    ids = []
    choices =  []
    queries = []
    answers = []
    vid_names = []
    with open(f'/shared/shang/datasets/nextqa/metadata/{filename}') as f:
        spamreader = csv.reader(f)
        for row in spamreader:
            #print(row)
            row = [x.strip() for x in row]
            sample_id,possible_answers,query_type,query,answer,video_name = row[1], row[2], row[3], row[4], row[5], row[6]
            ids.append(sample_id)
            choices.append(possible_answers)
            queries.append(query)
            answers.append(answer)
            vid_names.append(video_name)
    return ids, choices, queries, answers, vid_names
"""