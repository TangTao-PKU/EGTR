import os
import os.path as osp
from lib.core.config import BASE_DATA_DIR
from lib.models.smpl import SMPL, SMPL_MODEL_DIR
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import torch
import joblib
import shutil
import colorsys
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multi_person_tracker import MPT
from torch.utils.data import DataLoader
import subprocess
from collections import OrderedDict
import matplotlib.pyplot as plt

from lib.models.Graphformer import Graphformer
from lib.utils.renderer import Renderer
from lib.utils.demo_utils import (
    download_youtube_clip,
    convert_crop_cam_to_orig_img,
    CropDataset, 
    FeatureDataset
)

MIN_NUM_FRAMES = 25
random.seed(1)
torch.manual_seed(1)
np.random.seed(1)

# 视频转图像帧
def video_to_images(vid_file, img_folder=None):
    if img_folder is None:
        img_folder = osp.join('/home/public/tt/models/Graphformer_RELEASE/output', osp.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-r', '30',
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.jpg']
    print('运行视频生成图像帧的ffmpeg命令')
    #print(f'运行视频生成图像帧的ffmpeg命令 \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'保存图像帧至 \"{img_folder}\"')

    img_shape = cv2.imread(osp.join(img_folder, '000001.jpg')).shape

    return img_folder, len(os.listdir(img_folder)), img_shape

# 图像帧转视频
def images_to_video(img_folder, output_vid_file):
    os.makedirs(img_folder, exist_ok=True)

    command = [
        'ffmpeg', '-framerate', '30', '-y', '-threads', '16', '-i', f'{img_folder}/%06d.jpg', '-profile:v', 'baseline',
        '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', output_vid_file,
    ]
    print('运行图像帧生成视频的ffmpeg命令')
    #print(f'运行图像帧生成视频的ffmpeg命令\"{" ".join(command)}\"')
    subprocess.call(command)

def prepare_rendering_results(Graphformer_results, nframes):
    frame_results = [{} for _ in range(nframes)]
    for person_id, person_data in Graphformer_results.items():
        for idx, frame_id in enumerate(person_data['frame_ids']):
            frame_results[frame_id][person_id] = {
                'verts': person_data['verts'][idx],
                'cam': person_data['orig_cam'][idx],
                'bbox': person_data['bboxes'][idx],
            }

    # 基于弱透视相机尺度的朴素深度排序
    for frame_id, frame_data in enumerate(frame_results):
        # 根据原始图像坐标中y轴排序
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )
    return frame_results

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if(os.path.isdir('./output')):
        shutil.rmtree('./output')

    output_path = osp.join('./output/demo_output')
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 图像序列/视频输入
    if args.img_file:
        """ 准备输入图像序列 """
        image_folder = './Tdemo/images/change'
        num_frames = len(os.listdir(image_folder))
        #img_shape = cv2.imread(osp.join(image_folder, 'image_00000.jpg')).shape
        img_shape = cv2.imread(osp.join(image_folder, 'image_00532.jpg')).shape
    else:
        """ 准备输入视频 """
        video_file = args.vid_file
        if video_file.startswith('https://www.youtube.com'):
            print(f"正在下载网络视频 \'{video_file}\'")
            video_file = download_youtube_clip(video_file, './output/demo_output')
            if video_file is None:
                exit('url不存在')
            print(f"视频已保存在 {video_file}...")
        if not os.path.isfile(video_file):
            exit(f"输入视频 \'{video_file}\' 不存在!")
        image_folder, num_frames, img_shape = video_to_images(video_file)
        
    print(f"输入视频总帧数为:{num_frames}\n")
    orig_height, orig_width = img_shape[:2]

    """ 检测人体 """
    total_time = time.time()
    bbox_scale = 1.2
    mpt = MPT(
        device=device,
        batch_size=args.tracker_batch_size,
        display=args.display,
        detector_type=args.detector,
        output_format='dict',
        yolo_img_size=args.yolo_img_size,
    )
    # 调用MPT中__call__方法
    if args.save_mpt:
        tracking_results = mpt(image_folder,save=True)
    else:
        tracking_results = mpt(image_folder,save=False)
    # 数据格式{personid:{'bbox':[],'frames':[]}}

    # 如果帧数小于MIN_NUM_FRAMES，则删除tracklets
    for person_id in list(tracking_results.keys()):
        if tracking_results[person_id]['frames'].shape[0] < MIN_NUM_FRAMES:
            del tracking_results[person_id]

    """ 获取Graphformer模型 """
    seq_len = 16
    model = Graphformer(
        seqlen=seq_len,
        n_layers=2,
        hidden_size=1024
    ).to(device)

    # 加载预训练模型Graphformer
    pretrained_file = args.model
    ckpt = torch.load(pretrained_file)
    print(f"加载预训练模型\'{pretrained_file}\'")
    ckpt = ckpt['gen_state_dict']
    model.load_state_dict(ckpt, strict=False)

    # 必要时设置人体性别
    gender = args.gender  # 'neutral', 'male', 'female'
    model.regressor.smpl = SMPL(
        SMPL_MODEL_DIR,
        batch_size=64,
        create_transl=False,
        gender=gender
    ).cuda()
    model.eval()

    # 空间特征提取模型
    from lib.models.spin import hmr
    hmr = hmr().to(device)
    checkpoint = torch.load(osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'))
    hmr.load_state_dict(checkpoint['model'], strict=False)
    hmr.eval()

    """ 对每一个人运行Graphformer """
    print("\n在每个人体上运行Graphformer重建模型...")
    Graphformer_time = time.time()
    Graphformer_results = {}
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes =  joints2d = None
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        # 提取图像静态特征
        dataset = CropDataset(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )
        # dataset.bboxes.shape (122, 4)
        bboxes = dataset.bboxes
        frames = dataset.frames
        crop_dataloader = DataLoader(dataset, batch_size=256, num_workers=16)

        with torch.no_grad():
            feature_list = []
            for i, batch in enumerate(crop_dataloader):
                batch = batch.to(device)
                # 得到空间特征
                feature = hmr.feature_extractor(batch.reshape(-1,3,224,224))
                feature_list.append(feature.cpu())
            del batch

            feature_list = torch.cat(feature_list, dim=0)
            # torch.Size([122, 2048])
                    
            if(args.plotspatial!=-1):
                tokens = 16
                dimensions = 2048
                pos_encoding = feature_list[None,args.plotspatial-7:args.splotspatial+9,:]
                plt.figure(figsize=(12,8))
                plt.pcolormesh(pos_encoding[0], cmap='viridis')
                plt.xlabel('Spatial Dimensions')
                plt.xlim((0, dimensions))
                plt.ylim((tokens,0))
                plt.ylabel('Token Position')
                plt.colorbar()
                plt.savefig('./plot/SpatialFeatures.png')

        # 进行时序特征编码并估计人体网格
        dataset = FeatureDataset(
            image_folder=image_folder,
            frames=frames,
            seq_len=seq_len,
        )
        dataset.feature_list = feature_list
        #seq_list数量: 122
         
        dataloader = DataLoader(dataset, batch_size=64, num_workers=32)
        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], []

            for i, batch in enumerate(dataloader):
                batch = batch.to(device)
                # torch.Size([64, 16, 2048])
                # torch.Size([58, 16, 2048])
                # 用Graphformer得到smpl参数
                output = model(batch)[0][-1]

                pred_cam.append(output['theta'][:, :3])
                pred_verts.append(output['verts'])
                pred_pose.append(output['theta'][:, 3:75])
                pred_betas.append(output['theta'][:, 75:])
                pred_joints3d.append(output['kp_3d'])

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            del batch

        # ========= 保存结果至pkl文件 ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()

        bboxes[:, 2:] = bboxes[:, 2:] * 1.2
        if args.render_plain:
            pred_cam[:,0], pred_cam[:,1:] = 1, 0  
            # np.array([[1, 0, 0]])
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,   
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
        }
        # print(output_dict['verts'].shape)  (122, 6890, 3)
        Graphformer_results[person_id] = output_dict
    del model

    end = time.time()
    fps = num_frames / (end - Graphformer_time)
    print(f'重建帧率: {fps:.2f}')
    total_time = time.time() - total_time
    print(f'花费总时间(包含模型载入): {total_time:.2f} s')
    print(f'总帧率(包含模型载入): {num_frames / total_time:.2f}')

    if args.save_pkl:
        print(f"保存pkl结果至\'{os.path.join(output_path, 'Graphformer_output.pkl')}\'.")
        joblib.dump(Graphformer_results, os.path.join(output_path, "Graphformer_output.pkl"))

    """ 渲染人体网格并生成视频 """
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=args.wireframe)

    output_img_folder = f'{output_path}_render'
    input_img_folder = f'{output_path}_input'
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(input_img_folder, exist_ok=True)
    
    print(f"\n正在渲染重建结果, 正在将渲染帧写入{output_img_folder}")
    # 生成渲染数据
    #Graphformer_results[1]['verts'].shape (122, 6890, 3)
    frame_results = prepare_rendering_results(Graphformer_results, num_frames)
    #frame_results[1][1]['verts'].shape (6890, 3)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in Graphformer_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg')
    ])

    # 逐帧进行渲染
    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)
        input_img = img.copy()
        if args.render_plain:
            img[:] = 0
            # img[:] = 255

        if args.sideview:
            side_img = np.zeros_like(img)

        # 对帧中人体逐个渲染
        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mesh_filename = None
            if args.save_obj:
                mesh_folder = os.path.join(output_path, 'meshes', f'{person_id:02d}')
                Path(mesh_folder).mkdir(parents=True, exist_ok=True)
                mesh_filename = os.path.join(mesh_folder, f'{frame_idx:06d}.obj')

            mc = mesh_color[person_id]

            #进行网格重建
            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )
            if args.sideview:
                side_img = renderer.render(
                    side_img,
                    frame_verts,
                    cam=frame_cam,
                    color=mc,
                    angle=270,
                    axis=[0,1,0],
                )

        if args.sideview:
            # 将多个数组沿指定轴连接起来的函数
            img = np.concatenate([img, side_img], axis=1)

        # save output frames
        cv2.imwrite(os.path.join(output_img_folder, f'{frame_idx:06d}.jpg'), img)
        cv2.imwrite(os.path.join(input_img_folder, f'{frame_idx:06d}.jpg'), input_img)

        if args.display:
            cv2.imshow('Video', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if args.display:
        cv2.destroyAllWindows()

    """ 保存渲染后的视频 """
    #vid_name = os.path.basename(video_file)
    save_name = f'Graphformer_demo_output.mp4'
    save_path = os.path.join(output_path, save_name)

    images_to_video(img_folder=output_img_folder, output_vid_file=save_path)
    images_to_video(img_folder=input_img_folder, output_vid_file=os.path.join(output_path, 'demo_imput.mp4'))
    print(f"保存结果视频至{os.path.abspath(save_path)}")
    # shutil.rmtree(output_img_folder)
    shutil.rmtree(input_img_folder)
    # shutil.rmtree(segment_img_folder)
    # shutil.rmtree(image_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vid_file', type=str, default='./Tdemo/videos/demo.mp4', help='视频路径或链接')
    
    parser.add_argument('--img_file', action='store_true', help='图片序列路径')

    parser.add_argument('--model', type=str, default='./experiments/29-03-2023_23-12-12_GraphformerBEST/model_best.pth.tar', help='预训练模型')
    #parser.add_argument('--model', type=str, default='./data/base_data/tcmr_demo_model.pth.tar')
    #parser.add_argument('--model', type=str, default='./data/base_data/vibe_model_w_3dpw.pth.tar')
    #parser.add_argument('--model', type=str, default='./data/base_data/mpsnet_model_best.pth.tar')

    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo', 'maskrcnn'],
                        help='人体检测的检测器类型')

    parser.add_argument('--yolo_img_size', type=int, default=416,
                        help='yolo检测的输入图像大小')

    parser.add_argument('--tracker_batch_size', type=int, default=12,
                        help='人体检测的batchsize')

    parser.add_argument('--display', action='store_true',
                        help='可视化demo的每一步')
    
    parser.add_argument('--save_mpt', action='store_true',
                        help='是否保存人体检测图序列')

    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')
    
    parser.add_argument('--save_pkl', action='store_true',
                        help='保存pkl文件')
    
    parser.add_argument('--gender', type=str, default='neutral',
                        help='设置人体性别(neutral, male, female)')

    parser.add_argument('--wireframe', action='store_true',
                        help='用网格线渲染')

    parser.add_argument('--sideview', action='store_true',
                        help='从侧面视角渲染网格')

    parser.add_argument('--render_plain', action='store_true',
                        help='在全黑背景上渲染网格')

    parser.add_argument('--gpu', type=int, default='0', help='gpu序号')

    parser.add_argument('--plotspatial', type=int, default='-1',
                        help='绘制指定帧重建序列的空间特征')
    



    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    main(args)


# /home/public/anaconda3/envs/Graphformer-env/lib/python3.7/site-packages/torchvision/models/_utils.py:209: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.
#   f"The parameter '{pretrained_param}' is deprecated since 0.13 and may be removed in the future, "
# /home/public/anaconda3/envs/Graphformer-env/lib/python3.7/site-packages/torchvision/models/_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=ResNet50_Weights.IMAGENET1K_V1`. You can also use `weights=ResNet50_Weights.DEFAULT` to get the most up-to-date weights.
#   warnings.warn(msg)
