
# system
from __future__ import print_function, division
import os
from tqdm.notebook import tqdm

# common
import numpy as np
import pandas as pd

# opencv
import cv2

# torch
import torch
from torch.utils.data import Dataset
from torchvision import transforms


SPLIT_RATIO = 0.8

class FallDetectionDataset(Dataset):
    """FD Dataset"""
    
    # Standing: class 1, Sitting: class 2, Lying: class 3, Bending: class 4, Crawling: class 5, Empty: class 0
    
    def __init__(self, root_dir = 'data/FD/', transform=None, train=True, optical_flow=True,
                 frames_per_clip=10, step_between_clips=1, verbose=False, resize=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            train (bool, optional): if ``True``, creates a dataset from the train split,
                otherwise from the ``test`` split.
            frames_per_clip (int): number of frames in a clip.
            step_between_clips (int, optional): number of frames between each clip.
        Returns:
            video (Tensor[T, H, W, C]): the `T` video frames
            label (int): class of the video clip
        """
        self.root_dir = root_dir
        self.transform = transform
        self.verbose = verbose
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.optical_flow = optical_flow
        subdirs = [os.path.join(root_dir, item) for item in os.listdir(root_dir)
                   if os.path.isdir(os.path.join(root_dir, item))]
        
        if train:
            subdirs = subdirs[:int(len(subdirs)*SPLIT_RATIO)]
        else:
            subdirs = subdirs[int(len(subdirs)*SPLIT_RATIO):]
        
        self.video_clips = []
        self.targets = []
        for subdir in (subdirs):
            if self.verbose:
                print("[Dataset] reading folder: ", subdir)
            labels = pd.read_csv(os.path.join(subdir, "labels.csv"))
            labels.set_index('index', inplace=True)
            labels.loc[labels['class'] > 5, 'class'] = 0
            if resize:
                self.__resize_imgs(subdir)
            self.__create_optical_flow(subdir)
            clips, targets = self.__get_video_clips__(subdir, labels, frames_per_clip, step_between_clips)
            self.video_clips.extend(clips)
            self.targets.extend(targets)
    
    def __resize_imgs(self, video_dir):
        width, height=256, 256
        images_dir = os.path.join(video_dir, "rgb")
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        if self.verbose:
            print("[Dataset] resizing image files")
        for idx in tqdm(range(len(image_files)), disable=(not self.verbose)):
            curr_frame = cv2.imread(os.path.join(images_dir, image_files[idx]))
            curr_frame = cv2.resize(curr_frame, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(images_dir, image_files[idx]), curr_frame)
    
    def __create_optical_flow(self, video_dir):
        model = cv2.optflow.createOptFlow_Farneback()
        flows_dir = os.path.join(video_dir, "flow")
        if not os.path.isdir(flows_dir):
            os.mkdir(flows_dir)
        images_dir = os.path.join(video_dir, "rgb")
        image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
        prev_frame = None 
        prev_idx = -2
        if self.verbose:
            print("[Dataset] creating optical flows files")
        for idx in tqdm(range(1, len(image_files)), disable=(not self.verbose)):
            file_name = os.path.join(flows_dir, image_files[idx][:-4] + ".npz")
            if not os.path.isfile(file_name):
                if not prev_idx == idx-1:
                    prev_frame = cv2.imread(os.path.join(images_dir, image_files[idx-1]), 0)
                    prev_idx = idx - 1
                curr_frame = cv2.imread(os.path.join(images_dir, image_files[idx]), 0)
                flow = model.calc(prev_frame,curr_frame, None)
                np.savez_compressed(file_name, arr=flow.astype(np.float16))
                prev_frame, prev_idx = curr_frame, idx
        model.collectGarbage()

        
    def __save_clip__(self, data_dir, idx, clip_flows, clip_img):
        clip_dir = os.path.join(data_dir, 
                                "clip_" + str(self.frames_per_clip) + "_" + str(self.step_between_clips))
        if not os.path.isdir(clip_dir):
            os.mkdir(clip_dir)
        clip_flows = os.path.join(clip_dir, str(idx) + ".npz")
        if not os.path.isfile(clip_flows):
            flows = np.concatenate([np.load(file)['arr'] for file in clip_flows], axis=2)
            np.savez_compressed(clip_flows, flows)
        clip_obj = {
            "flows": clip_flows,
            "img": clip_img
        }
        return clip_obj
        
    def __get_video_clips__(self, data_dir, labels, frames_per_clip, step_between_clips=1):
        clips = []
        targets = []
        flows_dir = os.path.join(data_dir, "flow")
        imgs_dir = os.path.join(data_dir, "rgb")
        imgs = [os.path.join(imgs_dir, f) for f in os.listdir(imgs_dir) 
                     if os.path.isfile(os.path.join(imgs_dir, f))]
        flows = [os.path.join(flows_dir, f) for f in os.listdir(flows_dir) 
                     if os.path.isfile(os.path.join(flows_dir, f))]
        if self.verbose:
            print("[Dataset] creating video clips files")
        for idx in tqdm(range(0, len(imgs) - frames_per_clip, step_between_clips), disable=(not self.verbose)):
            # slice clip frames
            clip_flows = flows[idx:idx+frames_per_clip]
            clip_img = imgs[idx]
            clips.append(self.__save_clip__(data_dir, idx, clip_flows, clip_img))
            # get majority label
            indices = [int(f[-8:-4]) for f in clip_flows]
            frame_labels = list(labels.loc[indices]["class"])
            label = max(set(frame_labels), key = frame_labels.count)
            targets.append(label)
        return clips, targets
    
    
    def __len__(self):
        return len(self.video_clips)


    def __getitem__(self, idx):
        sample_file = self.video_clips[idx]
        if self.optical_flow:
            data = np.load(sample_file["flows"])["arr_0"].astype(np.float32)
        else:
            data = cv2.imread(sample_file["img"])
        if self.transform:
            data = self.transform(data)
            
        target = self.targets[idx]
        
        return data, target
