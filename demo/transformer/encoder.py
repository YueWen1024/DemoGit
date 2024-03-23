# coding=utf-8
# @FileName     : transformer.py
# @Time         : 2023/10/2 8:10
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import _get_clones
from multi_head_attention import MyMultiheadAttention


# 解码器层
class MyTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """
        :param d_model:         d_k = d_v = d_model/nhead = 64, 模型中向量的维度，论文默认值为 512
        :param nhead:           多头注意力机制中多头的数量，论文默认为值 8
        :param dim_feedforward: 全连接中向量的维度，论文默认值为 2048
        :param dropout:         丢弃率，论文中的默认值为 0.1
        """
        super(MyTransformerEncoderLayer, self).__init__()
        # 定义一个多头注意力机制模块
        self.self_attn = MyMultiheadAttention(d_model, nhead, dropout=dropout)
        # 定义其它层归一化和线性变换的模块
        self.dropout1 = nn.Dropout(dropout)  # 随机失活
        self.norm1 = nn.LayerNorm(d_model)  # 归一化
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.activation = F.relu
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param src_mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :param src_key_padding_mask:
        :return: # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, )[0]  # 计算多头注意力，src2: [src_len, batch_size, num_heads*kdim] num_heads*kdim = embed_dim
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # [src_len, batch_size, num_heads*kdim]

        src2 = self.activation(self.linear1(src))  # [src_len,batch_size,dim_feedforward]
        src2 = self.linear2(self.dropout(src2))  # [src_len,batch_size,num_heads*kdim]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src  # [src_len, batch_size, num_heads * kdim] <==> [src_len, batch_size, embed_dim]


# 编码器实现：堆叠多个编码层
class MyTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(MyTransformerEncoder, self).__init__()
        """
        encoder_layer: 就是包含有多头注意力机制的一个编码层
        num_layers: 克隆得到多个encoder layers 论文中默认为6
        norm: 归一化层
        """
        self.layers = _get_clones(encoder_layer, num_layers)  # self.layers中保存的便是一个包含有多个编码层的ModuleList
        # 克隆得到多个encoder layers 论文中默认为6
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        """
        :param src: 编码部分的输入，形状为 [src_len,batch_size, embed_dim]
        :param mask:  编码部分输入的padding情况，形状为 [batch_size, src_len]
        :return:# [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]
        """
        output = src
        for mod in self.layers:
            # 疑问：output的形状是什么样子？？？
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)  # 实现多个编码层堆叠起来的效果，并完成整个前向传播过程（多个encoder layers层堆叠后的前向传播过程）

        if self.norm is not None:
            output = self.norm(output)  # 对多个编码层的输出结果进行层归一化并返回最终的结果
        return output  # [src_len, batch_size, num_heads * kdim] <==> [src_len,batch_size,embed_dim]


# 编码器使用实例
if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    num_head = 3
    num_layers = 2
    src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim]
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [batch_size, src_len]

    my_transformer_encoder_layer = MyTransformerEncoderLayer(d_model=dmodel, nhead=num_head)
    my_transformer_encoder = MyTransformerEncoder(encoder_layer=my_transformer_encoder_layer,
                                                  num_layers=num_layers,
                                                  norm=nn.LayerNorm(dmodel))
    memory = my_transformer_encoder(src=src, mask=None, src_key_padding_mask=src_key_padding_mask)
    print(memory.shape)  # torch.Size([5, 2, 32])
