import random
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, DataLoader

from lib.dataset import *


class MultipleDatasets(Dataset):
    def __init__(self, dbs, make_same_len=True):
        self.dbs = dbs
        self.db_num = len(self.dbs)
        self.max_db_data_num = max([len(db) for db in dbs])
        self.db_len_cumsum = np.cumsum([len(db) for db in dbs])
        self.make_same_len = make_same_len

    def __len__(self):
        # all dbs have the same length
        if self.make_same_len:
            return self.max_db_data_num * self.db_num
        # each db has different length
        else:
            return sum([len(db) for db in self.dbs])

    def __getitem__(self, index):
        if self.make_same_len:
            db_idx = random.randint(0,self.db_num-1) # uniform sampling
            data_idx = index % self.max_db_data_num
            if data_idx >= len(self.dbs[db_idx]) * (self.max_db_data_num // len(self.dbs[db_idx])): # last batch: random sampling
                data_idx = random.randint(0,len(self.dbs[db_idx])-1)
            else: # before last batch: use modular
                data_idx = data_idx % len(self.dbs[db_idx])
        else:
            for i in range(self.db_num):
                if index < self.db_len_cumsum[i]:
                    db_idx = i
                    break
            if db_idx == 0:
                data_idx = index
            else:
                data_idx = index - self.db_len_cumsum[db_idx-1]

        return self.dbs[db_idx][data_idx]


def get_data_loaders(cfg):
    #重叠度是指相邻帧之间共同出现的物体的比例。15/16
    if cfg.TRAIN.OVERLAP:
        overlap = ((cfg.DATASET.SEQLEN-1)/float(cfg.DATASET.SEQLEN))
    else:
        overlap = 0

    def get_2d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(load_opt=cfg.TITLE,  seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)

    def get_3d_datasets(dataset_names):
        datasets = []
        for dataset_name in dataset_names:
            db = eval(dataset_name)(load_opt=cfg.TITLE, set='train', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
            datasets.append(db)
        return ConcatDataset(datasets)
        # 得到三个.pt文件的并集

    # ===== 2D keypoint datasets =====
    # 'Insta'
    train_2d_dataset_names = cfg.TRAIN.DATASETS_2D
    train_2d_db = get_2d_datasets(train_2d_dataset_names)

    data_2d_batch_size = int(cfg.TRAIN.BATCH_SIZE * cfg.TRAIN.DATA_2D_RATIO)
    data_3d_batch_size = cfg.TRAIN.BATCH_SIZE - data_2d_batch_size

    train_2d_loader = DataLoader(
        dataset=train_2d_db,
        batch_size=data_2d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== 3D keypoint datasets =====
    #  'ThreeDPW'
    # 'MPII3D'
    # 'Human36M'
    train_3d_dataset_names = cfg.TRAIN.DATASETS_3D
    train_3d_db = get_3d_datasets(train_3d_dataset_names)

    train_3d_loader = DataLoader(
        dataset=train_3d_db,
        batch_size=data_3d_batch_size,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
    )

    # ===== Evaluation dataset =====
    overlap = ((cfg.DATASET.SEQLEN-1)/float(cfg.DATASET.SEQLEN))
    valid_db = eval(cfg.TRAIN.DATASET_EVAL)(load_opt=cfg.TITLE, set='val', seqlen=cfg.DATASET.SEQLEN, overlap=overlap, debug=cfg.DEBUG)
    # valid_db.vid_indices = valid_db.vid_indices[::2]

    valid_loader = DataLoader(
        dataset=valid_db,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    return train_2d_loader, train_3d_loader, valid_loader

from lib.dataset import Dataset3D
from lib.core.config import H36M_DIR
from lib.core.config import MPII3D_DIR
from lib.core.config import THREEDPW_DIR
from lib.dataset import Dataset2D
from lib.core.config import PENNACTION_DIR
from lib.core.config import POSETRACK_DIR

class Human36M(Dataset3D):
    def __init__(self, load_opt, set, seqlen, overlap=0.75, debug=False):
        db_name = 'h36m'

        print('Human36M数据集相邻序列重叠率: ', overlap)
        super(Human36M, self).__init__(
            load_opt=load_opt,
            set=set,
            folder=H36M_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - 数据序列数量为 {self.__len__()}')

class MPII3D(Dataset3D):
    def __init__(self, load_opt, set, seqlen, overlap=0, debug=False):
        db_name = 'mpii3d'

        print('MPII3D数据集相邻序列重叠率: ', overlap)
        super(MPII3D, self).__init__(
            load_opt=load_opt,
            set = set,
            folder=MPII3D_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
        )
        print(f'{db_name} - 数据序列数量为 {self.__len__()}')
    
class ThreeDPW(Dataset3D):
    def __init__(self, load_opt, set, seqlen, overlap=0.75, debug=False, target_vid=''):
        db_name = '3dpw'

        print('3DPW数据集相邻序列重叠率: ', overlap)
        super(ThreeDPW, self).__init__(
            load_opt=load_opt,
            set=set,
            folder=THREEDPW_DIR,
            seqlen=seqlen,
            overlap=overlap,
            dataset_name=db_name,
            debug=debug,
            target_vid=target_vid
        )
        print(f'{db_name} - 数据序列数量为 {self.__len__()}')

class PennAction(Dataset2D):
    def __init__(self, load_opt, seqlen, overlap=0.75, debug=False):
        db_name = 'pennaction'

        super(PennAction, self).__init__(
            load_opt=load_opt,
            seqlen = seqlen,
            folder=PENNACTION_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - 数据序列数量为 {self.__len__()}')
        
class PoseTrack(Dataset2D):
    def __init__(self, load_opt, seqlen, overlap=0.75, folder=None, debug=False):
        db_name = 'posetrack'
        super(PoseTrack, self).__init__(
            load_opt=load_opt,
            seqlen = seqlen,
            folder=POSETRACK_DIR,
            dataset_name=db_name,
            debug=debug,
            overlap=overlap,
        )
        print(f'{db_name} - 数据序列数量为 {self.__len__()}')
        
import h5py
import torch
import logging
import numpy as np
import os.path as osp

from torch.utils.data import Dataset
from lib.core.config import Graphformer_DB_DIR
from lib.dataset._kp_utils import convert_kps
from lib.dataset._img_utils import normalize_2d_kp, split_into_chunks

logger = logging.getLogger(__name__)

class Insta(Dataset):
    def __init__(self, load_opt, seqlen, overlap=0., debug=False):
        self.seqlen = seqlen
        self.mid_frame = int(seqlen/2)
        self.stride = int(seqlen * (1-overlap) + 0.5)
        self.h5_file = osp.join(Graphformer_DB_DIR, 'insta_train_db.h5')

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db
            self.vid_indices = split_into_chunks(self.db['vid_name'], seqlen, self.stride)

        print(f'InstaVariety 数据序列数量为 {self.__len__()}')

    def __len__(self):
        return len(self.vid_indices)

    def __getitem__(self, index):
        return self.get_single_item(index)

    def get_sequence(self, start_index, end_index, data):
        if start_index != end_index:
            return data[start_index:end_index+1]
        else:
            return data[start_index:start_index+1].repeat(self.seqlen, axis=0)

    def get_single_item(self, index):
        start_index, end_index = self.vid_indices[index]

        with h5py.File(self.h5_file, 'r') as db:
            self.db = db

            kp_2d = self.get_sequence(start_index, end_index, self.db['joints2D'])
            kp_2d = convert_kps(kp_2d, src='insta', dst='spin')
            kp_2d_tensor = np.ones((self.seqlen, 49, 3), dtype=np.float16)

            input = torch.from_numpy(self.get_sequence(start_index, end_index, self.db['features'])).float()

            vid_name = self.get_sequence(start_index, end_index, self.db['vid_name'])
            frame_id = self.get_sequence(start_index, end_index, self.db['frame_id']).astype(str)
            instance_id = np.array([v.decode('ascii') + f for v, f in zip(vid_name, frame_id)])

        for idx in range(self.seqlen):
            kp_2d[idx,:,:2] = normalize_2d_kp(kp_2d[idx,:,:2], 224)
            kp_2d_tensor[idx] = kp_2d[idx]

        repeat_num = 3
        target = {
            'features': input,
            'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 2D keypoints transformed according to bbox cropping
            # 'instance_id': instance_id
        }

        return target

