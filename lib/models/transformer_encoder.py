import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as Fc

import math
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


class TransformerModel(nn.Module):
    def __init__(self,
                 seqlen = 16, 
                 input_size=2048,
                 d_model = 256):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.seqlen = seqlen
        # embed_dim = head_dim * num_heads?
        self.input_fc = nn.Linear(input_size, d_model)
        #self.output_fc = nn.Linear(input_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
            #device=device
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=5)
        

    def forward(self, x):
        y = x[:, -self. d_model:, :]
        x = self.input_fc(x)  # (256, 24, 128)
        x = self.pos_emb(x)   # (256, 24, 128)
        out = self.encoder(x)

        return out


def test_net():
    batch_size=32
    model = TransformerModel(seqlen = 16, input_size = 2048, d_model = 256) # 模型初始化
    model = model.cuda()
    model.eval()
    input = torch.randn(16,batch_size,2048).cuda() # 加载GPU
    # print("input:",input)
    # print('inputdim:',input.shape)    #[3, 16, 2048]
    #smpl_output1, scores1 = model(input) # 计算
    smpl_output1 = model(input)
    print(smpl_output1.shape)

if __name__ == '__main__':
    test_net() 