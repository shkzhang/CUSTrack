
import os
import shutil
from math import trunc

import matplotlib.pyplot as plt
import configparser

from networkx.utils import open_file
from tqdm import tqdm


def get_info(root_dir):
    info = {}
    gap = 10
    case_dirs = os.listdir(root_dir)
    case_num = len(case_dirs)
    info['case_num'] = case_num
    frame_num = 0
    x = []
    y = []
    video_list = []
    for case_dir in case_dirs:
        case_path = os.path.join(root_dir, case_dir)
        cur_frame_num = len(os.listdir(case_path)) - 5
        frame_num += cur_frame_num
        x.append(case_dir)
        y.append(cur_frame_num)
        config = configparser.ConfigParser()
        config.read(os.path.join(case_path, 'meta_info.ini'))
        value = config.get('METAINFO', 'url')
        if value not in video_list:
            video_list.append(value)
    sort_indices = sorted(range(len(y)), key=lambda k: y[k], reverse=True)
    x_sorted = [x[i] for i in sort_indices]
    y_sorted = [y[i] for i in sort_indices]
    plt.bar(x_sorted, y_sorted)
    plt.xlabel('X')
    plt.ylabel('Frames')
    plt.title('Label frame number')
    plt.xticks([])  # 不显示x坐标轴数字

    plt.savefig('frames.png')

    plt.show()
    info['avg_frame'] = frame_num / case_num
    info['frame'] = frame_num
    info['video_num'] = len(video_list)
    return info


def split_train_test(root_dir, split_dir):
    split_file = os.path.join(root_dir, 'test_split.txt')
    file = open(split_file, 'r', encoding='utf-8')
    test_case = file.readlines()
    test_case = [i.rstrip() for i in test_case]
    all_case = os.listdir(root_dir)
    os.makedirs(split_dir, exist_ok=True)
    train_dir =  os.path.join(split_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)
    test_dir =  os.path.join(split_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)
    train_file = open(os.path.join(train_dir,'list.txt'),'w',encoding='utf-8')
    test_file = open(os.path.join(test_dir,'list.txt'),'w',encoding='utf-8')

    for case in tqdm(all_case):
        case_path = os.path.join(root_dir, case)
        if not os.path.isdir(case_path): continue
        dest_path = train_dir
        list_file = train_file
        if case in test_case:
            dest_path = test_dir
            list_file = test_file
        list_file.write(case)
        list_file.write('\n')
        dest_path = os.path.join(dest_path, case)
        shutil.copytree(case_path,dest_path)
    train_file.close()
    test_file.close()


if __name__ == '__main__':
    # print(get_info(r"E:\Desktop\US-Liver-Track-Dataset"))
    split_train_test(r"/data1/object-track/ndth", r"/data1/object-track/ndth_clust_got10k")
