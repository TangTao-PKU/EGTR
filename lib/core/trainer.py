import time
import torch
import shutil
import logging
import numpy as np
import os.path as osp
from progress.bar import Bar

from lib.core.config import BASE_DATA_DIR
from lib.utils.utils import move_dict_to_device, AverageMeter, check_data_pararell

from lib.utils.eval_utils import (
    compute_accel,
    compute_error_accel,
    compute_error_verts,
    batch_compute_similarity_transform_torch,
)

logger = logging.getLogger(__name__)


class Trainer():
    def __init__(
            self,
            data_loaders,
            generator,
            gen_optimizer,
            end_epoch,
            criterion,
            start_epoch=0,
            lr_scheduler=None,
            device=None,
            writer=None,
            debug=False,
            debug_freq=1000,
            logdir='output',
            resume=None,
            performance_type='min',
            num_iters_per_epoch=1000,
    ):

        self.train_2d_loader, self.train_3d_loader, self.valid_loader = data_loaders

        self.train_2d_iter = self.train_3d_iter = None

        if self.train_2d_loader:
            self.train_2d_iter = iter(self.train_2d_loader)

        if self.train_3d_loader:
            # dataset重写item单个迭代器16*2048
            # datalodaer导入batchsize为32
            # 输入为 32*16*target
            self.train_3d_iter = iter(self.train_3d_loader)

        # 模型和优化器
        self.generator = generator
        self.gen_optimizer = gen_optimizer

        # 训练参数
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.writer = writer
        self.debug = debug
        self.debug_freq = debug_freq
        self.logdir = logdir

        self.performance_type = performance_type
        self.train_global_step = 0
        self.valid_global_step = 0
        self.epoch = 0
        self.best_performance = float('inf') if performance_type == 'min' else -float('inf')

        self.evaluation_accumulators = dict.fromkeys(['pred_j3d', 'target_j3d', 'target_theta', 'pred_verts'])

        self.num_iters_per_epoch = num_iters_per_epoch

        if self.writer is None:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=self.logdir)

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 从已训练的模型中恢复
        if resume is not None:
            self.resume_pretrained(resume)

    def train(self):
        # 一个epoch

        losses = AverageMeter()
        kp_2d_loss = AverageMeter()
        kp_3d_loss = AverageMeter()

        timer = {
            'data': 0,
            'forward': 0,
            'loss': 0,
            'backward': 0,
            'batch': 0,
        }

        self.generator.train()

        start = time.time()

        summary_string = ''

        bar = Bar(f'Epoch {self.epoch + 1}/{self.end_epoch}', fill='#', max=self.num_iters_per_epoch)
        time.sleep(6)
        # 共使用500*32=16000帧样本 并非用每个数据集的全部
        for i in range(self.num_iters_per_epoch):
            if(i%50==0):time.sleep(6)
            # 重置迭代器
            target_2d = target_3d = None
            if self.train_2d_iter:
                try:
                    target_2d = next(self.train_2d_iter)
                except StopIteration:
                    self.train_2d_iter = iter(self.train_2d_loader)
                    target_2d = next(self.train_2d_iter)

                move_dict_to_device(target_2d, self.device)

            if self.train_3d_iter:
                try:
                    # 数据格式如下
                    # batchsize * target
                    # target = {
                    #     'features': input,
                    #     'theta': torch.from_numpy(theta_tensor).float()[self.mid_frame].repeat(repeat_num, 1), # camera, pose and shape
                    #     'kp_2d': torch.from_numpy(kp_2d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 2D keypoints transformed according to bbox cropping
                    #     'kp_3d': torch.from_numpy(kp_3d_tensor).float()[self.mid_frame].repeat(repeat_num, 1, 1), # 3D keypoints
                    #     'w_smpl': w_smpl[self.mid_frame].repeat(repeat_num),
                    #     'w_3d': w_3d[self.mid_frame].repeat(repeat_num),
                    # }
                    target_3d = next(self.train_3d_iter)
                except StopIteration:
                    self.train_3d_iter = iter(self.train_3d_loader)
                    target_3d = next(self.train_3d_iter)

                move_dict_to_device(target_3d, self.device)


            # 前向生成器
            if target_2d and target_3d:
                inp = torch.cat((target_2d['features'], target_3d['features']), dim=0).cuda()
            elif target_3d:
                inp = target_3d['features'].cuda()
            else:
                inp = target_2d['features'].cuda()

            timer['data'] = time.time() - start
            start = time.time()

            preds, scores = self.generator(inp, is_train=True)

            timer['forward'] = time.time() - start
            start = time.time()

            gen_loss, loss_dict = self.criterion(
                generator_outputs=preds,
                data_2d=target_2d,
                data_3d=target_3d,
                scores=scores
            )

            timer['loss'] = time.time() - start
            start = time.time()

            # 反向生成器
            self.gen_optimizer.zero_grad()
            gen_loss.backward()
            self.gen_optimizer.step()


            # 训练日志
            total_loss = gen_loss
            losses.update(total_loss.item(), inp.size(0))
            kp_2d_loss.update(loss_dict['loss_kp_2d'].item(), inp.size(0)) #平均值
            kp_3d_loss.update(loss_dict['loss_kp_3d'].item(), inp.size(0))

            timer['backward'] = time.time() - start
            timer['batch'] = timer['data'] + timer['forward'] + timer['loss'] + timer['backward']
            start = time.time()

            summary_string = f'({i + 1}/{self.num_iters_per_epoch}) | 总耗时: {bar.elapsed_td} | ' \
                             f'预计剩余: {bar.eta_td:} | 总损失: {losses.avg:.2f} | 2d关节点平均损失: {kp_2d_loss.avg:.2f} ' \
                             f'| 3d关节点平均损失: {kp_3d_loss.avg:.2f} '

            for k, v in loss_dict.items():
                summary_string += f' | {k}: {v:.3f}'
                self.writer.add_scalar('train_loss/'+k, v, global_step=self.train_global_step)

            for k,v in timer.items():
                summary_string += f' | {k}: {v:.2f}'

            self.writer.add_scalar('train_loss/loss', total_loss.item(), global_step=self.train_global_step)

            if self.debug:
                print('==== Visualize ====')
                from lib.utils.vis import batch_visualize_vid_preds
                video = target_3d['video']
                dataset = 'spin'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target_3d.copy(),
                                                       vis_hmr=False, dataset=dataset)
                self.writer.add_video('train-video', vid_tensor, global_step=self.train_global_step, fps=10)

            self.train_global_step += 1
            bar.suffix = summary_string
            bar.next()

            if torch.isnan(total_loss):
                exit('损失无限小错误!...')

        bar.finish()

        logger.info(summary_string)

    def validate(self):
        self.generator.eval()
        time.sleep(6)

        start = time.time()

        summary_string = ''

        bar = Bar('验证', fill='#', max=len(self.valid_loader))

        if self.evaluation_accumulators is not None:
            for k,v in self.evaluation_accumulators.items():
                self.evaluation_accumulators[k] = []

        J_regressor = torch.from_numpy(np.load(osp.join(BASE_DATA_DIR, 'J_regressor_h36m.npy'))).float()

        for i, target in enumerate(self.valid_loader):
            if(i%100==0):time.sleep(6)
            move_dict_to_device(target, self.device)

            # <=============
            with torch.no_grad():
                inp = target['features']
                batch = len(inp)
                #print(inp.shape)  #[32, 16, 2048]

                preds, scores = self.generator(inp, J_regressor=J_regressor)
                #print(preds)
                #print(preds[-1]['kp_3d'].shape)     #[32,17,3] 正常17为14
                #print(scores.shape)                 #[32,3]

                # convert to 14 keypoint format for evaluation
                n_kp = preds[-1]['kp_3d'].shape[-2]
                #print(n_kp)                         #17
                #print(preds[-1]['kp_3d'].shape)     #[32,17,3]
                #print(target['kp_3d'].shape)        #[32,14,3]
                pred_j3d = preds[-1]['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                target_j3d = target['kp_3d'].view(-1, n_kp, 3).cpu().numpy()
                pred_verts = preds[-1]['verts'].view(-1, 6890, 3).cpu().numpy()
                target_theta = target['theta'].view(-1, 85).cpu().numpy()

                self.evaluation_accumulators['pred_verts'].append(pred_verts)
                self.evaluation_accumulators['target_theta'].append(target_theta)

                self.evaluation_accumulators['pred_j3d'].append(pred_j3d)
                self.evaluation_accumulators['target_j3d'].append(target_j3d)

            # DEBUG
            if self.debug and self.valid_global_step % self.debug_freq == 0:
                from lib.utils.vis import batch_visualize_vid_preds
                video = target['video']
                dataset = 'common'
                vid_tensor = batch_visualize_vid_preds(video, preds[-1], target, vis_hmr=False, dataset=dataset)
                self.writer.add_video('valid-video', vid_tensor, global_step=self.valid_global_step, fps=10)

            batch_time = time.time() - start

            summary_string = f'({i + 1}/{len(self.valid_loader)}) | batch: {batch_time * 10.0:.4}ms | ' \
                             f'总耗时: {bar.elapsed_td} | 预估剩余: {bar.eta_td:}'

            self.valid_global_step += 1
            bar.suffix = summary_string
            bar.next()

        bar.finish()

        logger.info(summary_string)

    def fit(self):

        for epoch in range(self.start_epoch, self.end_epoch):
            self.epoch = epoch
            self.train()
            self.validate()
            performance = self.evaluate()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(performance)


            # 记录学习率
            for param_group in self.gen_optimizer.param_groups:
                print(f'学习率 {param_group["lr"]}')
                self.writer.add_scalar('lr/gen_lr', param_group['lr'], global_step=self.epoch)

            logger.info(f'Epoch {epoch+1} 重建误差: {performance:.4f}')

            self.save_model(performance, epoch)

        self.writer.close()

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'gen_state_dict': self.generator.state_dict(),
            'performance': performance,
            'gen_optimizer': self.gen_optimizer.state_dict(),
        }

        filename = osp.join(self.logdir, 'checkpoint.pth.tar')
        torch.save(save_dict, filename)

        if self.performance_type == 'min':
            is_best = performance < self.best_performance
        else:
            is_best = performance > self.best_performance

        if is_best:
            logger.info('保存最好重建误差!')
            self.best_performance = performance
            shutil.copyfile(filename, osp.join(self.logdir, 'model_best.pth.tar'))

            with open(osp.join(self.logdir, 'best.txt'), 'w') as f:
                f.write(str(float(performance)))

    def resume_pretrained(self, model_path):
        if osp.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.best_performance = checkpoint['performance']

            logger.info(f"=> 加载已有模型 '{model_path}' "
                  f"(epoch {self.start_epoch}, performance {self.best_performance})")
        else:
            logger.info(f"=> 无已训练模型 '{model_path}'")

    def evaluate(self):

        for k, v in self.evaluation_accumulators.items():
            self.evaluation_accumulators[k] = np.vstack(v)

        pred_j3ds = self.evaluation_accumulators['pred_j3d']
        target_j3ds = self.evaluation_accumulators['target_j3d']

        pred_j3ds = torch.from_numpy(pred_j3ds).float()
        target_j3ds = torch.from_numpy(target_j3ds).float()

        print(f'在 {pred_j3ds.shape[0]} 个姿态上测试...')
        pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
        target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0

        pred_j3ds -= pred_pelvis
        target_j3ds -= target_pelvis

        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        pred_verts = self.evaluation_accumulators['pred_verts']
        target_theta = self.evaluation_accumulators['target_theta']

        m2mm = 1000

        pve = np.mean(compute_error_verts(target_theta=target_theta, pred_verts=pred_verts)) * m2mm
        accel = np.mean(compute_accel(pred_j3ds)) * m2mm
        accel_err = np.mean(compute_error_accel(joints_pred=pred_j3ds, joints_gt=target_j3ds)) * m2mm
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm

        eval_dict = {
            'mpjpe': mpjpe,
            'pa-mpjpe': pa_mpjpe,
            'accel': accel,
            'pve': pve,
            'accel_err': accel_err
        }

        log_str = f'Epoch {self.epoch+1}, '
        log_str += ' '.join([f'{k.upper()}: {v:.4f},'for k,v in eval_dict.items()])
        logger.info(log_str)

        for k,v in eval_dict.items():
            self.writer.add_scalar(f'error/{k}', v, global_step=self.epoch)

        return pa_mpjpe
