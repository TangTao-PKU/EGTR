import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor
import math


class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores

class ModulatedGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(ModulatedGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        self.adj = adj

        self.adj2 = nn.Parameter(torch.ones_like(adj))        
        nn.init.constant_(self.adj2, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')    

    def forward(self, input, is_train = False):

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])
        
        adj = self.adj.to(input.device) + self.adj2.to(input.device)
  
        adj = (adj.T + adj)/2
        
        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        
        output = torch.matmul(adj * E, self.M*h0) + torch.matmul(adj * (1 - E), self.M*h1)
        if self.bias is not None:
            output = output + self.bias.view(1, 1, -1)
        #32*2048*16
        return output

# ------------------------------#
# Positional Encoding的代码实现
# ------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=16):
        super(PositionalEncoding, self).__init__()

        # 位置编码的实现其实很简单，直接对照着公式去敲代码就可以，下面的代码只是其中的一种实现方式；
        # 从理解来讲，需要注意的就是偶数和奇数在公式上有一个共同部分，我们使用log函数把次方拿下来，方便计算；
        # pos代表的是单词在句子中的索引，这点需要注意；比如max_len是128个，那么索引就是从0，1，2，...,127
        # 假设我的d_model是512，2i以步长2从0取到了512，那么i对应取值就是0,1,2...255
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 这里需要注意的是pe[:, 0::2]这个用法，就是从0开始到最后面，步长为2，其实代表的就是偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 这里需要注意的是pe[:, 1::2]这个用法，就是从1开始到最后面，步长为2，其实代表的就是奇数位置

        # 下面这行代码之后，我们得到的pe形状是：[max_len * 1 * d_model]
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)  # 定一个缓冲区，其实简单理解为这个参数不更新就可以

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# ----------------------------#
# PoswiseFeedForwardNet的实现
# ----------------------------#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,in_channels=2048):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels=2048, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=2048, out_channels=in_channels, kernel_size=1)
        self.layer_norm = nn.LayerNorm(in_channels)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)

class TemporalEncoder(nn.Module):
    def __init__(
            self,
            n_layers=1,
            seq_len=16,
            hidden_size=2048
    ):
        super(TemporalEncoder, self).__init__()

        self.adj16 = nn.Parameter(torch.ones(16, 16))
        self.gcn16 = ModulatedGraphConv(2048,2048,adj=self.adj16)
        self.adj8 = nn.Parameter(torch.ones(8, 8))
        self.gcn8 = ModulatedGraphConv(2048,2048,adj=self.adj8)
        self.adj7 = nn.Parameter(torch.ones(7, 7))
        self.gcn7 = ModulatedGraphConv(2048,2048,adj=self.adj7)

        self.pe = PositionalEncoding(d_model=2048)
        self.trr = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1,padding=1, bias=False)
        self.ffn16 = PoswiseFeedForwardNet(in_channels=16)
        self.ffn8 = PoswiseFeedForwardNet(in_channels=8)
        self.ffn7 = PoswiseFeedForwardNet(in_channels=7)
        self.layer_norm = nn.LayerNorm(2048)
        self.getout16 = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1, stride=1,padding=0, bias=False)
        self.getout8 = nn.Conv1d(in_channels=8, out_channels=1, kernel_size=1, stride=1,padding=0, bias=False)
        self.getout7 = nn.Conv1d(in_channels=7, out_channels=1, kernel_size=1, stride=1,padding=0, bias=False)

        self.mid_frame = int(seq_len/2)
        self.hidden_size = hidden_size

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')

    def forward(self, x, is_train=False):
        
        # NTF
        x = self.trr(x)

        y_cur = self.pe(x.permute(1,0,2)).permute(1,0,2)

        x_cur = x
        x_bef = x[:, :self.mid_frame]
        x_aft = x[:, self.mid_frame+1:]
        y_cur = self.gcn16(y_cur) + x_cur
        y_cur = self.ffn16(y_cur.permute(0,2,1)).permute(0,2,1)
        y_cur = self.getout16(y_cur)

        y_bef = self.gcn8(x_bef) + x_bef
        y_bef = self.ffn8(y_bef.permute(0,2,1)).permute(0,2,1)
        y_bef = self.getout8(y_bef)

        y_aft = self.gcn7(x_aft) + x_aft
        y_aft = self.ffn7(y_aft.permute(0,2,1)).permute(0,2,1)
        y_aft = self.getout7(y_aft)
        # torch.Size([32, 16, 2048])
        # torch.Size([32, 7, 2048])
        # torch.Size([32, 8, 2048])
                
        y = torch.cat((y_bef, y_cur, y_aft), dim=1)
        #32*3*2048


        scores = self.attention(y)
        #32*3
        out = torch.mul(y, scores[:, :, None])
        out = torch.sum(out, dim=1)  # N x 2048

        if not is_train:
            return out, scores
        else:
            y = torch.cat((y[:, 0:1], y[:, 2:], out[:, None, :]), dim=1)
            return y, scores


class Graphformer(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(Graphformer, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.encoder = \
            TemporalEncoder(
                seq_len=seqlen,
                n_layers=n_layers,
                hidden_size=hidden_size
            )

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, is_train=False, J_regressor=None):
        # input size NTF
        batch_size, seqlen = input.shape[:2]

        feature, scores = self.encoder(input, is_train=is_train)
        feature = feature.reshape(-1, feature.size(-1))

        smpl_output = self.regressor(feature, is_train=is_train, J_regressor=J_regressor)
        #print(type(smpl_output))
        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores
        else:
            repeat_num = 3
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)
                s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3)
                s['scores'] = scores
        #print(smpl_output[-1]['kp_3d'].shape)#torch.Size([32, 3, 49, 3])
        return smpl_output, scores

def test_net():
    batch_size = 32
    model = TCMR(seqlen=16, batch_size=32, n_layers=1, hidden_size=1024) # 模型初始化
    model = model.cuda()
    model.eval()
    input = torch.randn(batch_size, 16, 2048).cuda() # 加载GPU
    # print("input:",input)
    # print('inputdim:',input.shape)    #[3, 16, 2048]
    smpl_output1, scores1 = model(input) # 计算
    # for s in smpl_output1:
    #     print(s['theta'].shape)
    #     #print(s['theta'])
    #     print(s['verts'].shape)
    #     #print(s['verts'])
    #     print(s['kp_2d'].shape)
    #     #print(s['kp_2d'])
    #     print(s['kp_3d'].shape)
    #     #print(s['kp_3d'])
    #     print(s['rotmat'].shape)
    #     #print(s['rotmat'])
    #     print(s['scores'].shape)
    #     #print(s['scores'])
    # print(scores1.shape)
    # #print(scores1)

if __name__ == '__main__':
    test_net() 


#export LD_LIBRARY_PATH=/home/public/anaconda3/envs/tcmr-env/lib/python3.7/site-packages/nvidia/cublas/lib export PYTHONPATH=$PYTHONPATH:/home/public/tt/models/TCMR_RELEASE/

# python demo.py --vid_file demo.mp4 --gpu 0 

# # dataset: 3dpw, mpii3d, h36m 
# python evaluate.py --dataset 3dpw --cfg ./configs/repr_table4_3dpw_model.yaml --gpu 0 

# # training outputs are saved in `experiments` directory
# # mkdir experiments
# python train.py --cfg ./configs/repr_table4_3dpw_model.yaml --gpu 0 

#watch -n 1 gpustat


#3DPW {'mpjpe': 85.84172*, 
# 'mpjpe_pa': 52.689568*, 
# 'accel_err': 7.887574062317824, 
# 'mpvpe': 101.0822*}
#mpii3d {'mpjpe': 99.79377, 
# 'mpjpe_pa': 62.510723*, 
# 'accel_err': 9.247442677941695}
#h36m {'mpjpe': 68.59867*, 
# 'mpjpe_pa': 47.539566*, 
# 'accel_err': 4.013048289206414}
