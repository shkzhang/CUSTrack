# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/6
# @Time        : 18:17
# @Description :
import os
import os.path

import cv2
import numpy as np
import torch
import csv
import pandas
import random
from collections import OrderedDict

from typing import Tuple

from .base_video_dataset import BaseVideoDataset
from core.train.data import jpeg4py_loader
from core.train.admin import env_settings


class CLUST(BaseVideoDataset):
    """ CLUST dataset.

    Publication:
        CLUST: https://clust.ethz.ch/results.html
    """

    def __init__(self, root=None, image_loader=jpeg4py_loader, split=None, seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the got-10k training data. Note: This should point to the 'train' folder inside GOT-10k
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            split - 'train' or 'val'. Note: The validation split here is a subset of the official got-10k train split,
                    not NOT the official got-10k validation split. To use the official validation split, provide that as
                    the root folder instead.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().clust_dir if root is None else root
        super().__init__("CLUST", root, image_loader)
        # all folders inside the root
        self.root = os.path.join(self.root, 'TrainingSet', '2D')

        self.sequence_list = self._get_sequence_list()
        self.all_frames = False

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            if seq_ids is not None:
                raise ValueError('Cannot set both split_name and seq_ids.')
            ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
            if split == 'train':
                file_path = os.path.join(ltr_path, 'data_specs', 'clust_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'data_specs', 'clust_test_split.txt')
            elif split == 'test':
                file_path = os.path.join(ltr_path, 'data_specs', 'clust_test_split.txt')
                self.all_frames = False
            else:
                raise ValueError('Unknown split name.')
            # seq_ids = pandas.read_csv(file_path, header=None, squeeze=True, dtype=np.int64).values.tolist()
            seq_ids = pandas.read_csv(file_path, header=None, dtype=str).squeeze("columns").values.tolist()
        elif seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [i for i in seq_ids]
        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list) * data_fraction))

        self.sequence_meta_info = self._load_meta_info()
        self.seq_per_class = self._build_seq_per_class()

        self.class_list = list(self.seq_per_class.keys())
        self.class_list.sort()
        self.add_weight_sample()


    def get_name(self):
        return 'clust'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return False

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _read_meta(self, seq_path):
        try:
            with open(os.path.join(seq_path, 'meta_info.ini')) as f:
                meta_info = f.readlines()
            object_meta = OrderedDict({'object_class_name': meta_info[5].split(': ')[-1][:-1],
                                       'motion_class': meta_info[6].split(': ')[-1][:-1],
                                       'major_class': meta_info[7].split(': ')[-1][:-1],
                                       'root_class': meta_info[8].split(': ')[-1][:-1],
                                       'motion_adverb': meta_info[9].split(': ')[-1][:-1]})
        except:
            object_meta = OrderedDict({'object_class_name': None,
                                       'motion_class': None,
                                       'major_class': None,
                                       'root_class': None,
                                       'motion_adverb': None})
        return object_meta

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class_name']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):
        seq_names = os.listdir(self.root)
        seq_anno_names = []
        for seq_name in seq_names:
            seq_dir = os.path.join(self.root, seq_name, 'Annotation')
            if not os.path.isdir(seq_dir): continue
            anno_files = os.listdir(seq_dir)
            anno_seq_names = []
            for i in anno_files:
                if not i.endswith('.txt'): continue
                anno_seq_name = os.path.splitext(i)[0]
                anno_seq_names.append(anno_seq_name)
            seq_anno_names += anno_seq_names
        seq_anno_names = sorted(seq_anno_names)
        with open(os.path.join(self.root, 'TrainingSet.txt'), 'w') as f:
            for seq_name in seq_anno_names:
                f.write(seq_name + '\n')
        return seq_anno_names

    def _split_seq_path_id(self, anno_seq_path) -> Tuple[str, str]:
        anno_seq_name = os.path.basename(anno_seq_path)
        return os.path.join(os.path.dirname(anno_seq_path), anno_seq_name.split('_')[0]), anno_seq_name

    def _from_center_to_xywh(self, anno, image, mask_size) -> np.ndarray:
        '''
          :param anno: 目标中心坐标(n,x,y)
          :param image: 原图
          :param mask_size: 目标ROI大小
          :return: 目标ROI框坐标(x1,y1,w,h)
           '''
        size = image.shape
        anno_mask = np.zeros((len(anno), 4))
        anno_mask[:, 0] = anno[:, 0] - mask_size / 2
        anno_mask[:, 0] = np.where(anno_mask[:, 0] > 0, anno_mask[:, 0], 0)

        anno_mask[:, 1] = anno[:, 1] - mask_size / 2
        anno_mask[:, 1] = np.where(anno_mask[:, 1] > 0, anno_mask[:, 1], 0)

        anno_mask[:, 2] = anno[:, 0] + mask_size / 2
        # anno_mask[:, 2] = np.where(anno_mask[:, 2] < size[0], anno_mask[:, 2], size[0])
        anno_mask[:, 2] = anno_mask[:, 2] - anno_mask[:, 0]

        anno_mask[:, 3] = anno[:, 1] + mask_size / 2
        # anno_mask[:, 3] = np.where(anno_mask[:, 3] < size[1], anno_mask[:, 3], size[1])
        anno_mask[:, 3] = anno_mask[:, 3] - anno_mask[:, 1]
        return anno_mask

    def _read_bb_anno(self, seq_path, mask_size=32):
        seq_path, class_name = self._split_seq_path_id(seq_path)
        bb_anno_file = os.path.join(seq_path, 'Annotation', f'{class_name}.txt')
        gt = np.loadtxt(bb_anno_file)
        img0 = cv2.imread(os.path.join(seq_path, 'Data', os.listdir(os.path.join(seq_path, 'Data'))[0]),
                          cv2.IMREAD_COLOR)
        box_gt = self._from_center_to_xywh(gt[:, 1:], img0, mask_size)
        return torch.tensor(gt[:, 0].reshape(-1, 1)), torch.tensor(box_gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover > 0).byte()

        visible_ratio = cover.float() / 8
        return target_visible, visible_ratio

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])
    def padding_read_bb_anno(self, seq_path, mask_size=32):
        seq_path, class_name = self._split_seq_path_id(seq_path)
        bb_anno_file = os.path.join(seq_path, 'Annotation', f'{class_name}.txt')
        gt = np.loadtxt(bb_anno_file)
        img0 = cv2.imread(os.path.join(seq_path, 'Data', os.listdir(os.path.join(seq_path, 'Data'))[0]),
                          cv2.IMREAD_COLOR)
        box_gt = self._from_center_to_xywh(gt[:, 1:], img0, mask_size)
        return torch.tensor(gt[:, 0].reshape(-1, 1)), torch.tensor(box_gt)
    def get_sequence_info(self, seq_id, mask_size=32):
        seq_path = self._get_sequence_path(seq_id)
        if self.all_frames:
            times, bbox = self.padding_read_bb_anno(seq_path, mask_size)
        else:
            times, bbox = self._read_bb_anno(seq_path, mask_size)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        # visible, visible_ratio = self._read_target_visible(seq_path)
        visible, visible_ratio = torch.ByteTensor([True]), torch.ByteTensor([0.1] * valid.shape[0])
        visible = visible & valid.byte()

        return {'bbox': bbox, 'times': times, 'valid': valid, 'visible': visible, 'visible_ratio': visible_ratio}

    def _find_frame_name(self, seq_path, frame_time):
        for file in os.listdir(seq_path):
            filename, fix = os.path.splitext(file)
            try:
                number = int(filename)
                if number == frame_time:
                    return file

            except ValueError:
                continue

    def _get_frame_path(self, seq_path, frame_id):
        return os.path.join(seq_path, self._find_frame_name(seq_path, frame_id))  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class_name']

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        mask_size = 32
        mask_size += np.random.randint(-8, 8)
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        # Read anno
        if anno is None:
            anno = self.get_sequence_info(seq_id, mask_size)

        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        # Get file ids
        anno_times = anno_frames['times']
        seq_path, class_name = self._split_seq_path_id(seq_path)

        frame_list = [self._get_frame(os.path.join(seq_path, 'Data'), f_id) for f_id in anno_times]
        return frame_list, anno_frames, obj_meta
