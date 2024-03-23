# coding=utf-8
# @FileName     : demo_loop.py
# @Time         : 2023/11/29 9:58
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 学习 Python 中的循环语句


def demo_while():
    # tgt, sums = 10, 0
    tgt, sums = 0, 10

    # 如果初始时，不满足循环的条件，则不会进入循环
    while sums >= tgt:
        print("进入循环体！")
        pass


if __name__ == '__main__':
    demo_while()
    pass
