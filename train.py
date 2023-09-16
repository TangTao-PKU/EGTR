import os
import os.path as osp
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import shutil

import torch
import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from lib.core.loss import GraphformerLoss
from lib.core.trainer import Trainer
from lib.core.config import parse_args, BASE_DATA_DIR
from lib.utils.utils import prepare_output_dir
from lib.models import Graphformer
from lib.dataset._loaders import get_data_loaders
from lib.utils.utils import create_logger, get_optimizer


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'实验的随机数种子 {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    # logger.info(f'GPU型号 -> {torch.cuda.get_device_name()}')
    # logger.info(f'GPU详细 -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn的相关配置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 为每次实验保存模型代码 便于版本管理
    output_model_dir = os.path.join(cfg.LOGDIR, 'Graphformer.py')
    shutil.copyfile(src='./lib/models/Graphformer.py', dst=output_model_dir)


    writer = SummaryWriter(log_dir=cfg.LOGDIR)
    writer.add_text('config', pprint.pformat(cfg), 0)

    # ========= 数据导入 ========= #
    data_loaders = get_data_loaders(cfg)

    # ========= 损失函数 ========= #
    loss = GraphformerLoss(
        e_loss_weight=cfg.LOSS.KP_2D_W,
        e_3d_loss_weight=cfg.LOSS.KP_3D_W,
        e_pose_loss_weight=cfg.LOSS.POSE_W,
        e_shape_loss_weight=cfg.LOSS.SHAPE_W,
    )

    # ========= 初始化网络、参数、学习率 ========= #
    generator = Graphformer(
        batch_size=cfg.TRAIN.BATCH_SIZE,
        seqlen=cfg.DATASET.SEQLEN,
        pretrained=cfg.TRAIN.PRETRAINED_REGRESSOR
    ).to(cfg.DEVICE)

    # 训练过程中更新的参数
    gen_optimizer = get_optimizer(
        model=generator,
        optim_type=cfg.TRAIN.GEN_OPTIM,
        lr=cfg.TRAIN.GEN_LR,
        weight_decay=cfg.TRAIN.GEN_WD,
        momentum=cfg.TRAIN.GEN_MOMENTUM,
    )

    # 学习率调整策略
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        gen_optimizer,
        mode='min',
        factor=0.1,
        patience=cfg.TRAIN.LR_PATIENCE,
        verbose=True,
    )
    
    # ========= 开始训练 ========= #
    Trainer(
        data_loaders=data_loaders,
        generator=generator,
        criterion=loss,
        gen_optimizer=gen_optimizer,
        start_epoch=cfg.TRAIN.START_EPOCH,
        end_epoch=cfg.TRAIN.END_EPOCH,
        device=cfg.DEVICE,
        writer=writer,
        debug=cfg.DEBUG,
        logdir=cfg.LOGDIR,
        lr_scheduler=lr_scheduler,
        resume=cfg.TRAIN.RESUME,
        num_iters_per_epoch=cfg.TRAIN.NUM_ITERS_PER_EPOCH,
        debug_freq=cfg.DEBUG_FREQ,
    ).fit()


if __name__ == '__main__':
    cfg, cfg_file, _ = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)