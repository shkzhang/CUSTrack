# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/6
# @Time        : 16:05
# @Description :
import os
import re

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import core.train.admin.settings as ws_settings

def clust_format(root_dir,output_dir):
    subset_dir_name = 'TrainingSet'
    subset_dir = os.path.join(root_dir, subset_dir_name,'2D')
    seq_names = os.listdir(subset_dir)
    seq_dirs = [os.path.join(subset_dir, s) for s in seq_names if os.path.isdir(os.path.join(subset_dir, s))]
    for seq_dir in tqdm(seq_dirs):
        for anno_file in os.listdir(os.path.join(seq_dir, 'Annotation')):
            one_output_dir = os.path.join(output_dir,anno_file.replace('.txt',''))
            img_files = sorted(os.listdir(os.path.join(seq_dir, 'Data')))
            img_files_time = sorted([int(re.findall(r'\d+', img_file)[0]) for img_file in img_files])
            if anno_file.endswith('.txt'):
                anno_path = os.path.join(seq_dir,'Annotation', anno_file)
                anno = np.loadtxt(anno_path)
                times = anno[:,0].astype(int)

                os.makedirs(one_output_dir,exist_ok=True)
                anno_list = []
                image_dir = os.path.join(one_output_dir,'imgs')
                os.makedirs(image_dir, exist_ok=True)
                for i,anno_image in enumerate(img_files):
                    if img_files_time[i] not in times:continue
                    one_anno = anno[np.where(img_files_time[i]==times)][0]
                    anno_image_path = os.path.join(seq_dir,'Data',anno_image)
                    image = Image.open(anno_image_path)
                    image = image.convert('RGB')
                    time = f'{img_files_time[i]:d}'
                    image.save(os.path.join(image_dir,f'{time}.png'))
                    anno_list.append(one_anno)

                np.savetxt(os.path.join(one_output_dir, 'label.txt'),np.stack(anno_list), fmt='%d',delimiter=',')

if __name__ == '__main__':
    data_dir = ws_settings.Settings().env.data_dir
    clust_format(os.path.join(data_dir,'clust'),os.path.join(data_dir,'clust_format'))
