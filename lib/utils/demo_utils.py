import os
import cv2
import time
import json
import torch
import subprocess
import numpy as np
import os.path as osp
from pytube import YouTube
from collections import OrderedDict

from lib.utils.smooth_bbox import get_smooth_bbox_params, get_all_bbox_params
from lib.dataset._img_utils import get_single_image_crop_demo

import os
import cv2
import numpy as np
import os.path as osp
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

from lib.utils.smooth_bbox import get_all_bbox_params
from lib.dataset._img_utils import get_single_image_crop_demo

# 得到裁剪后图片
class CropDataset(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size
        self.frames = frames
        # self.has_keypoints = True if joints2d is not None else False

        # self.norm_joints2d = np.zeros_like(self.joints2d)

        # if self.has_keypoints:
        #     bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        #     bboxes[:, 2:] = 150. / bboxes[:, 2:]
        #     self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

        #     self.image_file_names = self.image_file_names[time_pt1:time_pt2]
        #     self.joints2d = joints2d[time_pt1:time_pt2]
        #     self.frames = frames[time_pt1:time_pt2]

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img = cv2.cvtColor(cv2.imread(self.image_file_names[idx]), cv2.COLOR_BGR2RGB)

        bbox = self.bboxes[idx]

        # j2d = self.joints2d[idx] if self.has_keypoints else None
        j2d = None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=self.scale,
            crop_size=self.crop_size)
        # if self.has_keypoints:
        #     return norm_img, kp_2d
        # else:
        return norm_img

# 得到16帧序列
class FeatureDataset(Dataset):
    def __init__(self, image_folder, frames, seq_len=16):
        self.image_file_names = [
            osp.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.feature_list = None

        self.seq_len = seq_len
        self.seq_list = [[i, i+seq_len-1] for i in range(len(self.image_file_names) - seq_len + 1)]  # [i, i+seq_len-1] is inclusive
        for i in range(1, int(seq_len/2)+1):
            self.seq_list.insert(0, [int(seq_len/2)-i, int(seq_len/2)-i])
        for i in range(1, int(seq_len/2)):
            self.seq_list.append([-int(seq_len/2)+i, -int(seq_len/2)+i])

    def __len__(self):
        return len(self.seq_list)

    def get_sequence(self, start_index, end_index):
        if start_index != end_index:
            return self.feature_list[start_index:end_index+1]
        else:
            return self.feature_list[start_index][None,:].expand(self.seq_len,-1)

    def __getitem__(self, idx):
        start_idx, end_idx = self.seq_list[idx]

        return self.get_sequence(start_idx, end_idx)

def preprocess_video(video, joints2d, bboxes, frames, scale=1.0, crop_size=224):
    """
    Read video, do normalize and crop it according to the bounding box.
    If there are bounding box annotations, use them to crop the image.
    If no bounding box is specified but openpose detections are available, use them to get the bounding box.

    :param video (ndarray): input video
    :param joints2d (ndarray, NxJx3): openpose detections
    :param bboxes (ndarray, Nx5): bbox detections
    :param scale (float): bbox crop scaling factor
    :param crop_size (int): crop width and height
    :return: cropped video, cropped and normalized video, modified bboxes, modified joints2d
    """

    if joints2d is not None:
        bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
        bboxes[:,2:] = 150. / bboxes[:,2:]
        bboxes = np.stack([bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,2]]).T

        video = video[time_pt1:time_pt2]
        joints2d = joints2d[time_pt1:time_pt2]
        frames = frames[time_pt1:time_pt2]

    shape = video.shape

    temp_video = np.zeros((shape[0], crop_size, crop_size, shape[-1]))
    norm_video = torch.zeros(shape[0], shape[-1], crop_size, crop_size)

    for idx in range(video.shape[0]):

        img = video[idx]
        bbox = bboxes[idx]

        j2d = joints2d[idx] if joints2d is not None else None

        norm_img, raw_img, kp_2d = get_single_image_crop_demo(
            img,
            bbox,
            kp_2d=j2d,
            scale=scale,
            crop_size=crop_size)

        if joints2d is not None:
            joints2d[idx] = kp_2d

        temp_video[idx] = raw_img
        norm_video[idx] = norm_img

    temp_video = temp_video.astype(np.uint8)

    return temp_video, norm_video, bboxes, joints2d, frames

def trim_videos(filename, start_time, end_time, output_filename):
    command = ['ffmpeg',
               '-i', '"%s"' % filename,
               '-ss', str(start_time),
               '-t', str(end_time - start_time),
               '-c:v', 'libx264', '-c:a', 'copy',
               '-threads', '1',
               '-loglevel', 'panic',
               '"%s"' % output_filename]
    # command = ' '.join(command)
    subprocess.call(command)


def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = osp.join('/tmp', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', '30000/1001',
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.jpg']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.jpg')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


# 下载网络视频
def download_url(url, outdir):
    print(f'从{url}下载视频文件')
    cmd = ['wget', '-c', url, '-P', outdir]
    subprocess.call(cmd)

def download_youtube_clip(url, download_folder):
    return YouTube(url).streams.first().download(output_path=download_folder)

def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-framerate', '30000/1001', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.jpg', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]

    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)



def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    从裁剪后的图像坐标转换预测相机到原始图像坐标
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def prepare_rendering_results(Graphformer_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in Graphformer_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
                'bbox': person_data['bboxes'][idx],
            }

    # naive depth ordering based on the scale of the weak perspective camera
    # 基于弱透视相机尺度的朴素深度排序
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        # 根据原始图像坐标中y轴排序
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )
    return frame_results
