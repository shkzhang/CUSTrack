# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/28
# @Time        : 20:27
# @Description :
import os
import os.path

import numpy as np
import pandas

from core.test.evaluation.data import Sequence, BaseDataset, SequenceList
from core.test.utils.load_text import load_text
import os


class NDTHDataset(BaseDataset):
    """ NDTH dataset.
    """

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.root = self.env_settings.ndth_path
        self.sequence_list = self._get_sequence_list()
        self.size_threshold = 50
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'train')
        file_path = os.path.join(ltr_path, 'data_specs', 'ndth_test_split.txt')
        seq_ids = pandas.read_csv(file_path, header=None, dtype=str).squeeze("columns").values.tolist()

        if split == 'train':
            seq_ids = [i for i in self.sequence_list if i not in seq_ids]
        self.sequence_list = seq_ids
        if split == 'test_small':
            size = self.list_object_size()
            self.sequence_list = [seq for i, seq in enumerate(self.sequence_list) if size[i] <= self.size_threshold]
        if split == 'test_big':
            size = self.list_object_size()
            self.sequence_list = [seq for i, seq in enumerate(self.sequence_list) if size[i] > self.size_threshold]
        self.split = split

    def list_object_size(self):
        size = []
        for seq_name in self.sequence_list:
            size.append(self._construct_sequence_size(seq_name))
        return size

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence_size(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)
        ground_truth_rect = ground_truth_rect.reshape(-1, 4)
        return np.mean(np.sqrt(ground_truth_rect[:, 2] * ground_truth_rect[:, 3]))

    def _construct_sequence(self, sequence_name):
        anno_path = '{}/{}/groundtruth.txt'.format(self.base_path, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        frames_path = '{}/{}'.format(self.base_path, sequence_name)
        frame_list = [frame for frame in os.listdir(frames_path) if frame.endswith(".jpg")]
        frame_list.sort(key=lambda f: int(f[:-4]))
        frames_list = [os.path.join(frames_path, frame) for frame in frame_list]

        return Sequence(sequence_name, frames_list, 'ndth', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        return [i for i in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, i))]
