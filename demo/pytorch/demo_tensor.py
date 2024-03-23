# coding=utf-8
# @FileName     : demo_tensor.py
# @Time         : 2023/10/13 16:56
# @Author       : YueWen
# @Department   : AILAB
# @Description  :


import torch
import torch.nn.functional as F


def demo_view_trans():
    num_heads = 2  # 两个头
    head_dim = 64
    # 生成一个维度为(3, 4, 128)的随机张量 (bsz, s_len, num_heads*head_dim)
    query_states = torch.randn(3, 4, 128)  # 两个(1, 3, 128)
    bsz, s_len, _ = query_states.shape
    # print(query_states)  # 打印生成的张量

    query_states = query_states.view(bsz, s_len, num_heads, head_dim)
    print(query_states.shape)

    query_states = query_states.transpose(1, 2)
    print(query_states.shape)


def demo_expand():
    # mask 是一个二维的掩码张量，形状为 (tgt_len, tgt_len)
    # mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    # 生成一个维度为(3, 4)的随机张量 (bsz, s_len, num_heads*head_dim)
    tgt_len = 6
    mask = torch.randn(tgt_len, tgt_len)  # 6×6
    dtype = mask.dtype
    print(mask.shape)

    bsz = 2  # 设置两个批次
    past_key_values_length = 8
    if past_key_values_length > 0:  # 如果存在过去的键值对长度（通常在循环生成的情况下），则执行以下操作
        print('cat 前', mask.shape)
        # 在 mask 的左侧添加一个大小为 (tgt_len, past_key_values_length) 的全零张量，以处理过去的键值对。
        # 这是因为在循环生成的情况下，模型需要考虑之前生成的标记。
        # 6 × 6   6 × 8 -> 6 × 14
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask], dim=-1)
        print('cat 后', mask.shape)

    mask = mask[None, None, :, :]  # mask 张量添加两个维度，并且添加的两个维度都是1，[6, 6] -> [1, 1, 6, 6]
    print(mask.shape)
    # print(mask)
    mask = mask.expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
    print(mask.shape)
    # print(mask)  # 扩展的维度是直接复制相应维度上的内容


def demo_arange():
    tgt_len = 6
    dtype = torch.float32
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min)  #
    print(mask)

    mask_cond = torch.arange(mask.size(-1))  # 创建一个从0到tgt_len的整数数列一维张量（0, 1, 2, ···, tgt_len-1），不包tgt_len
    print(mask_cond)
    print((mask_cond + 1).view(mask.size(-1), 1))  # [6]
    # print((mask_cond + 1).view(mask.size(-1), 1).shape)  # [6] -> [6, 1]

    print(mask_cond < (mask_cond + 1).view(mask.size(-1), 1))

    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    print(mask)


def demo_broadcast():
    # 广播机制
    x = torch.ones(6)
    y = torch.ones(6, 1)

    # x比y少一个维度，所以x需要在其前面扩充一个维度，即[6] -> [1, 6]
    # x[1, 6]和y[6, 1]进行广播机制，从右到左开始
    # x的最后一个维度6和y的最后一个维度1广播，取6，即广播后的张量的最后一个维度是6
    # x的第一个维度1和y的第一个维度6广播，取较大值的6，即广播后的张量的第一个维度是6


def demo_unsqueeze():  # 升维
    """
    unsqueeze(): 指定维度上增加维度，该维度是1
    squeeze(): 删除指定的且维数为1的维度
    :return:
    """
    position_ids = torch.randn(3, 6)  # [bsz, s_len]
    print(position_ids.shape, position_ids)
    # position_ids = position_ids[:, -1]
    # print(position_ids.shape, position_ids)
    # position_ids = position_ids.unsqueeze(-1)
    # print(position_ids.shape, position_ids)
    position_ids = position_ids[:, -1].unsqueeze(-1)  # 对position_ids[:, -1]在最后一个维度上升维， [3] -> [3, 1]
    print(position_ids.shape, position_ids)


# 高纬
def demo_high_dim():
    torch.manual_seed(9)  # 设置随机种子
    q = torch.randn(4, 128)
    k = torch.randn(4, 128)

    dot = torch.matmul(q, k.transpose(0, 1))  # 使用torch.matmul进行点积计算
    print(dot)
    print(F.softmax(dot, dim=-1))
    print('—' * 200)

    q = torch.randn(4, 256)
    k = torch.randn(4, 256)
    dot = torch.matmul(q, k.transpose(0, 1))  # 使用torch.matmul进行点积计算
    print(dot)
    print(F.softmax(dot, dim=-1))
    print('—' * 200)

    q = torch.randn(4, 512)
    k = torch.randn(4, 512)
    dot = torch.matmul(q, k.transpose(0, 1))  # 使用torch.matmul进行点积计算
    print(dot)
    print(F.softmax(dot, dim=-1))
    print('—' * 200)

    q = torch.randn(4, 1024)
    k = torch.randn(4, 1024)
    dot = torch.matmul(q, k.transpose(0, 1))  # 使用torch.matmul进行点积计算
    print(dot)
    print(F.softmax(dot, dim=-1))
    print('—' * 200)

    q = torch.randn(4, 2048)
    k = torch.randn(4, 2048)
    dot = torch.matmul(q, k.transpose(0, 1))  # 使用torch.matmul进行点积计算
    print(dot)
    print(F.softmax(dot, dim=-1))

    pass


def demo_dtype_min():
    # torch.ones() 返回一个每个元素都是1、形状为size、数据类型为dtype的Tensor。
    bsz, s_len = 4, 30
    attention_mask = torch.ones((bsz, s_len), dtype=torch.bool)  # 返回形状为(bsz, s_len)、元素全为1、数据类型为torch.bool（元素全为True）的张量
    # print(torch.finfo(torch.bool).min)
    print(torch.bool.min())


if __name__ == '__main__':
    # demo_view_trans()
    # demo_expand()
    # demo_arange()
    # demo_broadcast()
    # demo_unsqueeze()
    # demo_high_dim()
    demo_dtype_min()
    pass
