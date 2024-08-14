from typing import Tuple

import cv2
import numpy as np
import pandas
import torch

from core.test.evaluation.data import Sequence, BaseDataset, SequenceList
from core.test.utils.load_text import load_text
import os


class CLUSTDataset(BaseDataset):
    """ CLUST dataset.

    Publication:

    """

    def __init__(self, split):
        super().__init__()
        # Split can be test, val, or ltrval (a validation split consisting of videos from the official train set)
        self.base_path = self.env_settings.clust_path
        self.split = split
        self.base_path = os.path.join(self.base_path, 'TrainingSet', '2D')
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'train')
        self.all_frames = False
        if split == 'train':
            file_path = os.path.join(ltr_path, 'data_specs', 'clust_train_split.txt')
        elif split == 'val':
            file_path = os.path.join(ltr_path, 'data_specs', 'clust_test_split.txt')
        else:
            file_path = os.path.join(ltr_path, 'data_specs', 'clust_test_split.txt')

        seq_ids = pandas.read_csv(file_path, header=None, dtype=str).squeeze("columns").values.tolist()
        self.sequence_list = seq_ids
        if split !='test':
            self.sequence_list = [i for i in self.sequence_list if i.split('-')[0]==split]

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

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

    def _split_seq_path_id(self, anno_seq_path) -> Tuple[str, str]:
        anno_seq_name = os.path.basename(anno_seq_path)
        return os.path.join(os.path.dirname(anno_seq_path), anno_seq_name.split('_')[0]), anno_seq_name

    def _read_bb_anno(self, seq_path: str, mask_size=32):
        if self.all_frames:
            return self.padding_read_bb_anno(seq_path, mask_size)

        seq_path, class_name = self._split_seq_path_id(seq_path)
        bb_anno_file = os.path.join(seq_path, 'Annotation', f'{class_name}.txt')
        gt = np.loadtxt(bb_anno_file)
        img0 = cv2.imread(os.path.join(seq_path, 'Data', os.listdir(os.path.join(seq_path, 'Data'))[0]),
                          cv2.IMREAD_COLOR)
        box_gt = self._from_center_to_xywh(gt[:, 1:], img0, mask_size)
        return np.float64(gt[:, 0].reshape(-1, 1)), np.float64(box_gt)

    def padding_read_bb_anno(self, seq_path, mask_size=32):
        seq_path, class_name = self._split_seq_path_id(seq_path)
        bb_anno_file = os.path.join(seq_path, 'Annotation', f'{class_name}.txt')
        gt = np.loadtxt(bb_anno_file)
        img0 = cv2.imread(os.path.join(seq_path, 'Data', os.listdir(os.path.join(seq_path, 'Data'))[0]),
                          cv2.IMREAD_COLOR)
        box_gt = self._from_center_to_xywh(gt[:, 1:], img0, mask_size)
        times = gt[:, 0]
        max_time = int(np.max(times))
        min_time = int(np.min(times))

        new_time = []
        new_anno = []
        box_gt_index = 0
        for i in range(min_time,max_time+1):
            new_time.append(i)
            if (times == i).any():
                new_anno.append(box_gt[box_gt_index])
                box_gt_index += 1
            else:
                new_anno.append(np.array([-1, -1, -1, -1]))

        new_times = np.array(new_time)
        new_anno = np.array(new_anno)
        return np.float64(new_times.reshape(-1, 1)), np.float64(new_anno)

    def _construct_sequence(self, sequence_name: str):
        sequence_path_name = sequence_name.split('_')[0]
        times, ground_truth_rect = self._read_bb_anno(os.path.join(str(self.base_path), sequence_name))

        frames_path = os.path.join(self.base_path, sequence_path_name, 'Data')
        frame_list = os.listdir(frames_path)
        frame_list.sort(key=lambda f: int(os.path.splitext(f)[0]))

        frames_list = [os.path.join(frames_path, frame) for frame in frame_list if
                       float(os.path.splitext(frame)[0]) in times]

        return Sequence(sequence_name, frames_list, 'clust', ground_truth_rect.reshape(-1, 4))

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self, split):
        with open('{}/list.txt'.format(self.base_path)) as f:
            sequence_list = f.read().splitlines()

        if split == 'ltrval':
            with open('{}/got10k_val_split.txt'.format(self.env_settings.dataspec_path)) as f:
                seq_ids = f.read().splitlines()

            sequence_list = [sequence_list[int(x)] for x in seq_ids]
        return sequence_list
