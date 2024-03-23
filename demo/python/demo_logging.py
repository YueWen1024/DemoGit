# coding=utf-8
# @FileName     : demo_logging.py
# @Time         : 2023/11/23 11:16
# @Author       : YueWen
# @Department   : AILAB
# @Description  : 学习 Python logging（Python 标准库）
import logging
import sys


# logging库中的级别对应值
def demo_level_value():
    print("DEBUG:", logging.DEBUG)
    print("INFO:", logging.INFO)
    print("WARNING:", logging.WARNING)
    print("ERROR:", logging.ERROR)
    print("CRITICAL:", logging.CRITICAL)


def demo_logging_base():
    # 此处进行Logging.basicConfig() 设置，后面设置无效
    logging.basicConfig(level=logging.ERROR)  # 配置日志的输出，如配置输出级别：代码会将level以上级别日志输出
    logging.debug(' 用来打印一些调试信息，等级最低！')
    logging.info(' 用来打印一些正常的操作信息！')
    logging.warning(' 用来用来打印警告信息！')
    logging.error(' 用来打印一些错误信息！')
    logging.critical(' 用来打印一些致命的错误信息，等级最高！')


def demo_logging_base_01():
    # 此处进行Logging.basicConfig() 设置，后面设置无效
    # 设置了filename后，那么就不会在控制台输出信息了，直接写入文件中
    logging.basicConfig(filename="test.log",
                        filemode="w",
                        encoding='UTF-8',
                        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                        datefmt="%d-%M-%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)],
                        level=logging.ERROR)
    logging.debug(' 用来打印一些调试信息，等级最低！')
    logging.info(' 用来打印一些正常的操作信息！')
    logging.warning(' 用来用来打印警告信息！')
    logging.error(' 用来打印一些错误信息！')
    logging.critical(' 用来打印一些致命的错误信息，等级最高！')


if __name__ == '__main__':
    # demo_level_value()
    # demo_logging_base()
    demo_logging_base_01()
    pass
