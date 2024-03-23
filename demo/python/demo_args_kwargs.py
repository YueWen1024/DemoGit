# coding=utf-8
# @FileName     : demo_args_kwargs.py
# @Time         : 2023/11/29 9:23
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 学习 Python 函数中的不定长参数（*args 和 **kwargs 参数）

# 参数
# Python调用函数时可使用的正式参数类型：
# 必需参数
# 关键字参数
# 默认参数
# 不定长参数
# *的参数会以元组(tuple)的形式导入，存放所有未命名的变量参数。
# **的参数会以字典的形式导入

def demo_arg(arg1, *args):
    print("输出: ")
    print(arg1)
    print(type(args), args)  # args是元组类型


def demo_run_args():
    demo_arg(1, 1, 2)


def demo_kwarg(arg1, **kwargs):
    print("输出: ")
    print(arg1)
    print(type(kwargs), kwargs)  # kwargs是字典类型


def demo_run_kwargs():
    demo_kwarg(1, a=1, b=2)


def demo_args_kwargs(arg1, *args, kwarg1="default_value", **kwargs):
    print("arg1:", arg1)
    print("*args:", args)
    print("kwarg1:", kwarg1)
    print("**kwargs:", kwargs)


def demo_run():
    # 调用示例
    demo_args_kwargs(1, 2, 3, kwarg1="custom_value", key1="value1", key2="value2")


if __name__ == '__main__':
    # demo_run_args()
    # demo_run_kwargs()
    pass
