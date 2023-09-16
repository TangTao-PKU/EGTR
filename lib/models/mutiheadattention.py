"""
original from:
https://github.com/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


# ----------------------------------#
# get_attn_subsequent_mask的实现
# ----------------------------------#
def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


# ----------------------------------#
# ScaledDotProductAttention的实现
# ----------------------------------#
class ScaledDotProductAttention(nn.Module):
    def __init__(self,d_k  = 64):
        self.d_k = d_k
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        # 输入进来的维度分别是 [batch_size x n_heads x len_q x d_k]  K： [batch_size x n_heads x len_k x d_k]
        # V: [batch_size x n_heads x len_k x d_v]
        # 首先经过matmul函数得到的scores形状是: [batch_size x n_heads x len_q x len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)

        # 然后最关键的地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对其他单词就不会起作用
        # scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context, attn


# -------------------------#
# MultiHeadAttention的实现
# -------------------------#
class MultiHeadAttention(nn.Module):
    def __init__(self,
            d_model = 512,
            d_ff = 2048 ,
            d_k  = 64 ,
            d_v = 64,
            n_layers = 6 ,
            n_heads = 8):
        super(MultiHeadAttention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.d_model = d_model
        self.d_ff = d_ff
        self.d_v = d_v
        self.d_k = d_k
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V):

        # 这个多头注意力机制分为这几个步骤，首先映射分头，然后计算atten_scores，然后计算atten_value;
        # 输入进来的数据形状： Q: [batch_size x len_q x d_model], K: [batch_size x len_k x d_model],
        # V: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # 下面这个就是先映射，后分头；一定要注意的是q和k分头之后维度是一致额，所以一看这里都是dk
        # q_s: [batch_size x n_heads x len_q x d_k]
        q_s = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # k_s: [batch_size x n_heads x len_k x d_k]
        k_s = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        # v_s: [batch_size x n_heads x len_k x d_v]
        v_s = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)

        # 输入进来的attn_mask形状是batch_size x len_q x len_k，然后经过下面这个代码得到
        # 新的attn_mask: [batch_size x n_heads x len_q x len_k]，就是把pad信息重复到了n个头上
        # attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)

        # 然后我们运行ScaledDotProductAttention这个函数
        # 得到的结果有两个：context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q x len_k]
        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s,)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


# ----------------------------#
# PoswiseFeedForwardNet的实现
# ----------------------------#
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self,
            d_model = 512,
            d_ff = 2048 ,):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


# --------------------------#
# get_attn_pad_mask的实现：
# --------------------------#
# 比如说，我现在的句子长度是5，在后面注意力机制的部分，我们在计算出来QK转置除以根号之后，softmax之前，我们得到的形状len_input * len*input
# 代表每个单词对其余包含自己的单词的影响力。所以这里我需要有一个同等大小形状的矩阵，告诉我哪个位置是PAD部分，之后在计算softmax之前会把这里置
# 为无穷大；一定需要注意的是这里得到的矩阵形状是batch_size x len_q x len_k，我们是对k中的pad符号进行标识，并没有对k中的做标识，因为没必要。
# seq_q和seq_k不一定一致，在交互注意力，q来自解码端，k来自编码端，所以告诉模型编码这边的pad符号信息就可以，解码端的pad信息在交互注意力层是
# 没有用到的；
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


# ------------------------------#
# Positional Encoding的代码实现
# ------------------------------#
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=16):
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


# ---------------------------------------------------#
# EncoderLayer：包含两个部分，多头注意力机制和前馈神经网络
# ---------------------------------------------------#
class EncoderLayer(nn.Module):
    def __init__(self,
            d_model = 512):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()
        self.pos_emb = PositionalEncoding(d_model)
        self.conv_att = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1,padding=0, bias=False)

    def forward(self, enc_inputs):
        """
        下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model]，需要注意的是最初始的QKV矩阵是等同于这个
        输入的，去看一下enc_self_attn函数.
        """

        enc_inputs = self.pos_emb(enc_inputs.transpose(0, 1)).transpose(0, 1)
        # enc_inputs to same Q,K,V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)
        # enc_outputs: [batch_size x len_q x d_model]
        enc_outputs = self.pos_ffn(enc_outputs)
        attn = self.conv_att(attn)[:,0,:,:]
        return enc_outputs, attn


# # -----------------------------------------------------------------------------#
# # Encoder部分包含三个部分：词向量embedding，位置编码部分，自注意力层及后续的前馈神经网络
# # -----------------------------------------------------------------------------#
# class Encoder(nn.Module):
#     def __init__(self,
#             d_model = 512,
#             n_layers = 6):
#         super(Encoder, self).__init__()
#         # 这行其实就是生成一个矩阵，大小是: src_vocab_size * d_model
#         # self.src_emb = nn.Embedding(src_vocab_size, d_model)
#         # 位置编码，这里是固定的正余弦函数，也可以使用类似词向量的nn.Embedding获得一个可以更新学习的位置编码
#         self.pos_emb = PositionalEncoding(d_model)
#         # 使用ModuleList对多个encoder进行堆叠，因为后续的encoder并没有使用词向量和位置编码，所以抽离出来；
#         self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

#     def forward(self, enc_inputs):
#         """
#         这里我们的enc_inputs形状是： [batch_size x source_len]
#         """

#         # 下面这行代码通过src_emb进行索引定位，enc_outputs输出形状是[batch_size, src_len, d_model]
#         #enc_outputs = self.src_emb(enc_inputs)
#         enc_outputs = enc_inputs

#         # 这行是位置编码，把两者相加放到了pos_emb函数里面
#         enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)

#         # get_attn_pad_mask是为了得到句子中pad的位置信息，给到模型后面，在计算自注意力和交互注意力的时候去掉pad符号的影响
#         # enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)
#         enc_self_attns = []
#         for layer in self.layers:
#             # 去看EncoderLayer层函数
#             enc_outputs, enc_self_attn = layer(enc_outputs)
#             enc_self_attns.append(enc_self_attn)
#         return enc_outputs, enc_self_attns



if __name__ == '__main__':

    # # 句子的输入部分:
    # # ich mochte ein bier P: 编码端的输入，P代表填充字符Pad
    # # S i want a beer: 解码端的输入，S代表Start
    # # i want a beer E: 解码端的真实标签，E代表End
    # sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # # ------------------------------------------------------------------------------#
    # # 一些简单的配置文件：
    # # Transformer Parameters
    # # Padding Should be Zero
    # # 构建词表：
    # # 构建编码端词表
    # src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    # src_vocab_size = len(src_vocab)

    # # 构建解码端词表，其实编码端和解码端可以共用一个词表
    # tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    # tgt_vocab_size = len(tgt_vocab)

    # src_len = 5  # 输入长度
    # tgt_len = 5  # 解码端输入长度

    # # 模型参数
    # d_model = 512  # 每个字符转换为Embedding的时候的大小
    # d_ff = 2048  # 前馈神经网络中Linear层映射到多少维度
    # d_k = d_v = 64  # dimension of K(=Q), V
    # n_layers = 6  # 6个encoder
    # n_heads = 8  # 多头注意力机制的时候把我的头分为几个
    # # ------------------------------------------------------------------------------#

    # 模型部分
    model = EncoderLayer()
    model = model.cuda()
    model.eval()

    batch_size=32
    
    input = torch.randn(batch_size, 16, 512).cuda() # 加载GPU
    out,att= model(input)
    print(out.shape)
    print(att.shape)
