# coding=utf-8
# @FileName     : demo_multi.py
# @Time         : 2024/1/23 9:33
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 向量、矩阵之间的点成、叉乘，广播机制
import torch


def demo_dot():
    a = torch.tensor([2, 3])
    b = torch.tensor([-1, 3])
    res = torch.dot(a, b)  # 2×（-1）+3×3
    print(res)

    a = torch.tensor([2, 3, 5])
    b = torch.tensor([-1, 3, 5])
    res = torch.dot(a, b)  # 2×（-1）+3×3+5×5
    print(res)


if __name__ == '__main__':
    demo_dot()
    pass
