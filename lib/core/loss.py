import torch
import torch.nn as nn
import numpy as np
from lib.utils.geometry import batch_rodrigues

def perm_index_reverse(indices):
    indices_reverse = np.copy(indices)

    for i, j in enumerate(indices):
        indices_reverse[j] = i

    return indices_reverse


class GraphformerLoss(nn.Module):
    def __init__(
            self,
            e_loss_weight=300.,
            e_3d_loss_weight=300.,
            e_pose_loss_weight=60.,
            e_shape_loss_weight=6.,
            device='cuda',
    ):
        super(GraphformerLoss, self).__init__()
        self.e_loss_weight = e_loss_weight
        self.e_3d_loss_weight = e_3d_loss_weight
        self.e_pose_loss_weight = e_pose_loss_weight
        self.e_shape_loss_weight = e_shape_loss_weight

        self.device = device
        #self.criterion_shape = nn.L1Loss().to(self.device)
        self.criterion_keypoints = nn.MSELoss(reduction='none').to(self.device)
        self.criterion_regr = nn.MSELoss().to(self.device)
        self.criterion_attention = nn.CrossEntropyLoss()


    def forward(
            self,
            generator_outputs,
            data_2d,
            data_3d,
            scores,
            data_body_mosh=None,
            data_motion_mosh=None,
            body_discriminator=None,
            motion_discriminator=None,
    ):
        # 减少时间维度
        reduce = lambda x: x.contiguous().view((x.shape[0] * x.shape[1],) + x.shape[2:])
        # reshape权重向量
        flatten = lambda x: x.reshape(-1)
        # 累积所有预测的 thetas
        accumulate_thetas = lambda x: torch.cat([output['theta'] for output in x],0)

        if data_2d:
            sample_2d_count = data_2d['kp_2d'].shape[0]
            real_2d = torch.cat((data_2d['kp_2d'], data_3d['kp_2d']), 0)
        else:
            sample_2d_count = 0
            real_2d = data_3d['kp_2d']

        real_2d = reduce(real_2d)
        real_3d = reduce(data_3d['kp_3d'])
        data_3d_theta = reduce(data_3d['theta'])

        w_3d = data_3d['w_3d'].type(torch.bool)
        w_smpl = data_3d['w_smpl'].type(torch.bool)

        total_predict_thetas = accumulate_thetas(generator_outputs)

        preds = generator_outputs[-1]
        pred_j3d = preds['kp_3d'][sample_2d_count:]
        pred_theta = preds['theta'][sample_2d_count:]

        theta_size = pred_theta.shape[:2]

        pred_theta = reduce(pred_theta)
        pred_j2d = reduce(preds['kp_2d'])
        pred_j3d = reduce(pred_j3d)

        w_3d = flatten(w_3d)
        w_smpl = flatten(w_smpl)

        pred_theta = pred_theta[w_smpl]
        pred_j3d = pred_j3d[w_3d]
        data_3d_theta = data_3d_theta[w_smpl]
        real_3d = real_3d[w_3d]

        # 损失函数
        loss_kp_2d = self.keypoint_loss(pred_j2d, real_2d, openpose_weight=1., gt_weight=1.) * self.e_loss_weight
        loss_kp_3d = self.keypoint_3d_loss(pred_j3d, real_3d)
        loss_kp_3d = loss_kp_3d * self.e_3d_loss_weight

        real_shape, pred_shape = data_3d_theta[:, 75:], pred_theta[:, 75:]
        real_pose, pred_pose = data_3d_theta[:, 3:75], pred_theta[:, 3:75]

        loss_dict = {
            'loss_kp_2d': loss_kp_2d,
            'loss_kp_3d': loss_kp_3d,
        }
        if pred_theta.shape[0] > 0:
            loss_pose, loss_shape = self.smpl_losses(pred_pose, pred_shape, real_pose, real_shape)
            loss_shape = loss_shape * self.e_shape_loss_weight
            loss_pose = loss_pose * self.e_pose_loss_weight
            loss_dict['loss_shape'] = loss_shape
            loss_dict['loss_pose'] = loss_pose
            #print(self.e_shape_loss_weight)

        gen_loss = torch.stack(list(loss_dict.values())).sum()

        return gen_loss, loss_dict

    def attention_loss(self, pred_scores):
        gt_labels = torch.ones(len(pred_scores)).to(self.device).long()

        return -0.5 * self.criterion_attention(pred_scores, gt_labels)

    def accel_3d_loss(self, pred_accel, gt_accel):
        pred_accel = pred_accel[:, 25:39]
        gt_accel = gt_accel[:, 25:39]

        if len(gt_accel) > 0:
            return self.criterion_accel(pred_accel, gt_accel).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def keypoint_loss(self, pred_keypoints_2d, gt_keypoints_2d, openpose_weight, gt_weight):
        """
        计算关键点上的 2D 重投影损失。
        损失由置信度加权。
        每个数据集的可用关键点都不同。
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        conf[:, :25] *= openpose_weight
        conf[:, 25:] *= gt_weight
        loss = (conf * self.criterion_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
        return loss

    def keypoint_3d_loss(self, pred_keypoints_3d, gt_keypoints_3d):
        """
        为 3D 关键点注释可用的示例计算 3D 关键点损失。
        损失由置信度加权。
        """
        pred_keypoints_3d = pred_keypoints_3d[:, 25:39, :]
        gt_keypoints_3d = gt_keypoints_3d[:, 25:39, :]
        # conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        # gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
        # gt_keypoints_3d = gt_keypoints_3d
        # conf = conf
        pred_keypoints_3d = pred_keypoints_3d
        if len(gt_keypoints_3d) > 0:
            gt_pelvis = (gt_keypoints_3d[:, 2,:] + gt_keypoints_3d[:, 3,:]) / 2
            gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
            pred_pelvis = (pred_keypoints_3d[:, 2,:] + pred_keypoints_3d[:, 3,:]) / 2
            pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
            # print(conf.shape, pred_keypoints_3d.shape, gt_keypoints_3d.shape)
            # return (conf * self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()
            return self.criterion_keypoints(pred_keypoints_3d, gt_keypoints_3d).mean()
        else:
            return torch.FloatTensor(1).fill_(0.).to(self.device)

    def smpl_losses(self, pred_rotmat, pred_betas, gt_pose, gt_betas):
        pred_rotmat_valid = batch_rodrigues(pred_rotmat.reshape(-1,3)).reshape(-1, 24, 3, 3)
        gt_rotmat_valid = batch_rodrigues(gt_pose.reshape(-1,3)).reshape(-1, 24, 3, 3)
        pred_betas_valid = pred_betas
        gt_betas_valid = gt_betas
        if len(pred_rotmat_valid) > 0:
            loss_regr_pose = self.criterion_regr(pred_rotmat_valid, gt_rotmat_valid)
            loss_regr_betas = self.criterion_regr(pred_betas_valid, gt_betas_valid)
        else:
            loss_regr_pose = torch.FloatTensor(1).fill_(0.).to(self.device)
            loss_regr_betas = torch.FloatTensor(1).fill_(0.).to(self.device)
        return loss_regr_pose, loss_regr_betas
