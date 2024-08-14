
import os

import cv2
import numpy as np
from tqdm import tqdm


def _from_center_to_xywh(anno, image, width=32, height=32) -> np.ndarray:
    '''
      :param anno: 目标中心坐标(n,x,y)
      :param image: 原图
      :param mask_size: 目标ROI大小
      :return: 目标ROI框坐标(x1,y1,w,h)
       '''
    size = image.shape
    anno_mask = np.zeros((len(anno), 4))
    anno_mask[:, 0] = anno[:, 0] - width / 2
    anno_mask[:, 0] = np.where(anno_mask[:, 0] > 0, anno_mask[:, 0], 0)  # x1

    anno_mask[:, 1] = anno[:, 1] - height / 2
    anno_mask[:, 1] = np.where(anno_mask[:, 1] > 0, anno_mask[:, 1], 0)  # y1

    anno_mask[:, 2] = anno[:, 0] + width / 2
    # anno_mask[:, 2] = np.where(anno_mask[:, 2] < size[0], anno_mask[:, 2], size[0])
    anno_mask[:, 2] = anno_mask[:, 2] - anno_mask[:, 0]

    anno_mask[:, 3] = anno[:, 1] + height / 2
    # anno_mask[:, 3] = np.where(anno_mask[:, 3] < size[1], anno_mask[:, 3], size[1])
    anno_mask[:, 3] = anno_mask[:, 3] - anno_mask[:, 1]
    return anno_mask


def get_frame(frame_id, images_path):
    for file in images_path:
        filename, fix = os.path.splitext(os.path.basename(file))
        try:
            number = int(filename)
            if number == frame_id:
                return file

        except ValueError:
            print("Error: %s is not a number" % filename)
            continue


def get_anno(frame_file, time_list, anno_list):
    filename, fix = os.path.splitext(os.path.basename(frame_file))
    frame = int(filename)
    for frame_id, anno in zip(time_list, anno_list):
        if frame == frame_id:
            return anno


def clust_format(root_path, case_name, width, height, output_path, with_all=False, show_label=False):
    root_path = os.path.join(root_path, 'TrainingSet', '2D')
    output_path = os.path.join(output_path, case_name)
    os.makedirs(output_path, exist_ok=False)
    case_dir = case_name.split('_')[0]
    case_dir = os.path.join(root_path, case_dir)
    anno_dir = os.path.join(case_dir, 'Annotation')
    images_dir = os.path.join(case_dir, 'Data')
    images = os.listdir(images_dir)
    images_path = [os.path.join(images_dir, i) for i in images]
    anno_file = os.path.join(anno_dir, f'{case_name}.txt')
    anno = np.loadtxt(anno_file)
    anno = np.array(anno)
    img0 = cv2.imread(images_path[0])
    times = anno[:, 0]
    anno_xywh = _from_center_to_xywh(anno[:, 1:], img0, width, height)

    video_info = {
        "url": case_name,
        "begin": 0,
        "end": len(images_path),
        "anno_fps": "4Hz",
        "object_class": "markers",
        "motion_class": "breathing",
        "major_class": "blood vessel or lesion",
        "root_class": "clust_markers",
        "motion_adverb": "slowly",
        "resolution": f"({img0.shape[1]}, {img0.shape[0]})"
    }

    with open(os.path.join(output_path, 'meta_info.ini'), 'w') as file:
        file.write("[METAINFO]\n")
        for key, value in video_info.items():
            file.write("{}: {}\n".format(key, value))
    absence = open(os.path.join(output_path, 'absence.label'), 'w')
    cover = open(os.path.join(output_path, 'cover.label'), 'w')
    cut_by_image = open(os.path.join(output_path, 'cut_by_image.label'), 'w')
    groundtruth = open(os.path.join(output_path, 'groundtruth.txt'), 'w')
    if not with_all:
        for label_id, (frame, one_anno) in enumerate(zip(times, anno_xywh)):
            x, y, w, h = one_anno.astype(int)[0:4]
            frame_file = get_frame(frame, images_path)
            assert frame_file is not None
            image = cv2.imread(frame_file)
            height, width, layers = image.shape
            if show_label:
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imwrite(os.path.join(output_path, '{:08d}.jpg'.format(label_id + 1)), image)
            absence.write('0\n')
            cover.write(str(int(10 * (w * h) / (height * width))) + '\n')
            cut_by_image.write('0\n')
            groundtruth.write(f'{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n')
    elif with_all:
        label_id = 0
        for frame in images_path:
            image = cv2.imread(frame)
            anno = get_anno(frame, times, anno_xywh)
            if anno is None and label_id==0:continue
            height, width, layers = image.shape
            absence.write('0\n')
            cut_by_image.write('0\n')
            if anno is not None:
                x, y, w, h = anno.astype(int)[0:4]
                cover.write(str(int(10 * (w * h) / (height * width))) + '\n')
                if show_label:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                groundtruth.write(f'{x:.4f},{y:.4f},{w:.4f},{h:.4f}\n')
            else:
                cover.write('1\n')
                groundtruth.write(f'{-1:.4f},{-1:.4f},{-1:.4f},{-1:.4f}\n')
            cv2.imwrite(os.path.join(output_path, '{:08d}.jpg'.format(label_id + 1)), image)
            label_id+=1
    cover.close()
    absence.close()
    cut_by_image.close()
    groundtruth.close()


def add_clust_all(src_path, output_path):
    with open(os.path.join(src_path, 'test_split.txt'), 'r', encoding='utf-8') as f:
        for case_name in tqdm(f.readlines()):
            case_name = case_name.rstrip()
            clust_format(src_path, case_name, 32, 32, os.path.join(output_path, 'test'), with_all=False, show_label=False)
    with open(os.path.join(src_path, 'train_split.txt'), 'r', encoding='utf-8') as f:
        for case_name in tqdm(f.readlines()):
            case_name = case_name.rstrip()
            clust_format(src_path, case_name, 32, 32, os.path.join(output_path, 'train'), with_all=False,
                         show_label=False)
    train_file = open(os.path.join(output_path, 'train', 'list.txt'), 'w', encoding='utf-8')
    test_file = open(os.path.join(output_path, 'test', 'list.txt'), 'w', encoding='utf-8')

    for case in os.listdir(os.path.join(output_path, 'train')):
        case_path = os.path.join(output_path, 'train', case)
        if not os.path.isdir(case_path): continue
        train_file.write(case)
        train_file.write('\n')

    for case in os.listdir(os.path.join(output_path, 'test')):
        case_path = os.path.join(output_path, 'test', case)
        if not os.path.isdir(case_path): continue
        test_file.write(case)
        test_file.write('\n')
    train_file.close()
    test_file.close()


if __name__ == '__main__':
    # clust_format(r"E:\dataset\clust", 'CIL-01_1', 44, 16,
    #              r'E:\dataset\clust_format\Train',
    #              show_label=True, with_all=True)
    add_clust_all(r'/data1/object-track/clust_2d_train', r'/data1/object-track/ndth_clust_got10k')
