# coding=utf-8
# @FileName     : transformer.py
# @Time         : 2023/9/26 8:10
# @Author       : YueWen
# @Department   : AILAB
# @Description  :
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.modules.transformer  # pytorch中的Transformer源码
from torch.nn import Parameter


# 多头注意力前向传播 仔细观察一下各个变量维度的变化过程
def multi_head_attention_forward(
    query,  # [tgt_len, batch_size, embed_dim]  query、key、value 均是指没有经过线性变换前的输入
    key,  # [src_len, batch_size, embed_dim]
    value,  # [src_len, batch_size, embed_dim]
    num_heads,
    dropout_p,
    out_proj_weight,  # [embed_dim = vdim * num_heads, embed_dim]
    out_proj_bias,
    training=True,
    key_padding_mask=None,  # [batch_size,src_len/tgt_len]
    q_proj_weight=None,  # [embed_dim, kdim * num_heads]
    k_proj_weight=None,  # [embed_dim, kdim * num_heads]
    v_proj_weight=None,  # [embed_dim, vdim * num_heads]
    attn_mask=None,  # [tgt_len,src_len]
  ):
    """
    query、key、value 均是指没有经过线性变换前的输入：
      - 在编码时三者指的均是原始输入序列src；
      - 在解码时的Mask Multi-Head Attention中三者指的均是目标输入序列tgt；
      - 在解码时的Encoder-Decoder Attention中三者分别指的是Mask Multi-Head Attention的输出、Memory（Encoder的输出）和Memory（Encoder的输出）。

    :param query:
    :param key:
    :param value:
    :param num_heads:
    :param dropout_p:
    :param out_proj_weight:
    :param out_proj_bias:
    :param training:
    :param key_padding_mask: 指编码或解码部分，输入序列的Padding情况，形状为[batch_size,src_len]或者[batch_size,tgt_len]
    :param q_proj_weight: q的权重矩阵
    :param k_proj_weight: k的权重矩阵
    :param v_proj_weight: v的权重矩阵
    :param attn_mask: 注意力掩码矩阵，形状为[tgt_len,src_len]，它只会在解码时使用。
    :return:
    """
    # 第一阶段： 计算得到Q、K、V
    q = F.linear(query, q_proj_weight)
    # [tgt_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [tgt_len, batch_size, kdim * num_heads]
    k = F.linear(key, k_proj_weight)
    # [src_len, batch_size, embed_dim] x [embed_dim, kdim * num_heads] = [src_len,batch_size, kdim * num_heads]
    v = F.linear(value, v_proj_weight)
    # [src_len, batch_size, embed_dim] x [embed_dim, vdim * num_heads] = [src_len, batch_size, vdim * num_heads]

    # 第二阶段： 缩放，以及attn_mask维度判断
    tgt_len, bsz, embed_dim = query.size()  # [tgt_len, batch_size, embed_dim]
    src_len = key.size(0)  # tgt_len本质上指的其实是query_len；src_len本质上指的是key_len。只是在不同情况下两者可能会是一样，也可能会是不一样 ？？？存在疑问
    head_dim = embed_dim // num_heads  # 每个头的维度 num_heads * head_dim = embed_dim
    scaling = float(head_dim) ** -0.5  # 缩放系数，根下d_k分之一
    q = q * scaling  # [query_len, batch_size, kdim * num_heads]  (Q*K^T)/√d_k = K^T*(Q/√d_k) = Q*(K^T/√d_k)
    if attn_mask is not None:  # 用来判断或修改attn_mask的维度，当然这几行代码只会在解码器中的Masked Multi-Head Attention中用到
        # [tgt_len, src_len] or [num_heads * batch_size, tgt_len, src_len]
        if attn_mask.dim() == 2:  # 此时attn_mask的形状为[tgt_len,src_len]
            attn_mask = attn_mask.unsqueeze(0)  # [1, tgt_len,src_len] 扩充维度
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        # 现在 atten_mask 的维度就变成了3D

    # 第三阶段： 计算得到注意力权重矩阵
    # contiguous()：用于确保张量的内存是连续的
    # view(tgt_len, bsz * num_heads, head_dim)：将q张量形状重新整理为一个新的三维(tgt_len, bsz * num_heads, head_dim)张量  ？？？存在疑问
    # tgt_len：表示目标序列的长度。
    # bsz * num_heads：表示批量大小（batch size）乘以注意力头的数量（num_heads），即每个头的查询都被堆叠在一起。（个人添加：是不是可以理解为每个头都是同级别？）
    # head_dim：表示每个注意力头的维度。
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads, tgt_len, kdim] 因为前面是num_heads个头一起参与的计算，所以这里要进行一下变形，以便于后面计算。且同时交换了0，1两个维度
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads, src_len, kdim]
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    # [batch_size * num_heads, src_len, vdim]
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))  # num_heads个QK相乘后的注意力矩阵 此时的q是经过缩放后的q  bmm的作用是用来计算两个三维矩阵的乘法操作
    # [batch_size * num_heads, tgt_len, kdim] x [batch_size * num_heads, kdim, src_len] =  [batch_size * num_heads, tgt_len, src_len]

    # 第四阶段： 进行相关掩码操作
    if attn_mask is not None:  # 要遮掩的位置设置无穷小，对应位置相加后的结果仍是无穷小
        attn_output_weights += attn_mask  # [batch_size * num_heads, tgt_len, src_len]
    if key_padding_mask is not None:
        # 变成 [batch_size, num_heads, tgt_len, src_len]的形状
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        # 扩展维度，从[batch_size, src_len]变成[batch_size, 1, 1, src_len]
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))  # 进行Padding Mask，遮掩位置使用无穷小替换
        # [batch_size * num_heads, tgt_len, src_len]
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    # [batch_size * num_heads, tgt_len, src_len]
    attn_output_weights = F.softmax(attn_output_weights, dim=-1)  # 无穷小（遮掩）部分被置为0
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)  # 随机失活
    # Z = [batch_size * num_heads, tgt_len, src_len]  x  [batch_size * num_heads, src_len, vdim] = [batch_size * num_heads, tgt_len, vdim]
    attn_output = torch.bmm(attn_output_weights, v)  # num_heads个Attention(Q, K, V)结果  attn_output的维度是什么样子的？

    # transpose(0, 1): 0维和1维进行交换 [tgt_len, batch_size * num_heads, kdim]，再view成[tgt_len, batch_size, num_heads * kdim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)

    Z = F.linear(attn_output, out_proj_weight, out_proj_bias)
    # 这里就是多个z 线性组合成Z [tgt_len, batch_size, embed_dim]
    return Z, attn_output_weights.sum(dim=1) / num_heads  # 将num_heads个注意力权重矩阵按对应维度取平均


# 多头注意力
class MyMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True):
        super(MyMultiheadAttention, self).__init__()
        """
        :param embed_dim:   词嵌入的维度，也就是前面的d_model参数，论文中的默认值为512
        :param num_heads:   多头注意力机制中多头的数量，也就是前面的nhead参数， 论文默认值为 8
        :param bias:        最后对多头的注意力（组合）输出进行线性变换时，是否使用偏置
        """
        self.embed_dim = embed_dim  # 前面的d_model参数
        self.head_dim = embed_dim // num_heads  # head_dim 指的就是d_k,d_v
        self.kdim = self.head_dim
        self.vdim = self.head_dim
        self.num_heads = num_heads  # 多头个数
        self.dropout = dropout
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim 除以 num_heads必须为整数"
        # 上面的限制条件就是论文中的  d_k = d_v = d_model/n_head 条件
        self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        # embed_dim = kdim * num_heads
        # 这里第二个维度之所以是embed_dim，实际上这里是同时初始化了num_heads个W_q堆叠起来的, 也就是num_heads个头
        self.k_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        # W_k,  embed_dim = kdim * num_heads
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
        # W_v,  embed_dim = vdim * num_heads
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 最后将所有的Z组合起来的时候，也是一次性完成， embed_dim = vdim * num_heads

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        """
        在论文中，编码时query, key, value 都是同一个输入，
        解码时 输入的部分也都是同一个输入，
        解码和编码交互时 key,value指的是 memory, query指的是tgt
        :param query: # [tgt_len, batch_size, embed_dim], tgt_len 表示目标序列的长度
        :param key:  #  [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param value: # [src_len, batch_size, embed_dim], src_len 表示源序列的长度
        :param attn_mask: # [tgt_len,src_len] or [num_heads*batch_size,tgt_len, src_len]
                一般只在解码时使用，为了并行一次喂入所有解码部分的输入，所以要用mask来进行掩盖当前时刻之后的位置信息
        :param key_padding_mask: [batch_size, src_len], src_len 表示源序列的长度
        :return:
        attn_output: [tgt_len, batch_size, embed_dim]
        attn_output_weights: # [batch_size, tgt_len, src_len]
        """
        return multi_head_attention_forward(query, key, value, self.num_heads,
                                            self.dropout, self.out_proj.weight, self.out_proj.bias,
                                            training=self.training,
                                            key_padding_mask=key_padding_mask,
                                            q_proj_weight=self.q_proj_weight,
                                            k_proj_weight=self.k_proj_weight,
                                            v_proj_weight=self.v_proj_weight,
                                            attn_mask=attn_mask)


if __name__ == '__main__':
    src_len = 5
    batch_size = 2
    dmodel = 32
    num_head = 1
    src = torch.rand((src_len, batch_size, dmodel))  # shape: [src_len, batch_size, embed_dim] [5, 2, 32]
    print(src)
    src_key_padding_mask = torch.tensor([[True, True, True, False, False],
                                         [True, True, True, True, False]])  # shape: [src_len, src_len]

    my_mh = MyMultiheadAttention(embed_dim=dmodel, num_heads=num_head)
    r = my_mh(src, src, src, key_padding_mask=src_key_padding_mask)  # 表示Q、K、V都是输入词向量 -> (带有注意力的V, 注意力权重)
    print(r)
    # print(res)
    # print(type(res), len(res))
    # print(res[0], res[1])