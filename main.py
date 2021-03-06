# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import pandas as pd
import numpy as np
from math import log

kp = pd.read_csv('kp.csv', index_col=0)
ab = pd.read_csv('alphaBeta.csv', index_col=0)
sheet5 = pd.read_csv('sheet5.csv', index_col=0)


def calc_design_storm(table, prob, cv, h):
    col_p = table[prob]
    hp = np.interp(cv, col_p.index, col_p.values) * h
    return hp


def calc_hour_rainfall(table, hp, f):
    hour = np.arange(1, 25).tolist()

    # 内插点雨量
    h1p, h6p, h24p = hp[0], hp[1], hp[2]
    hour_hp = [h1p]
    for i in range(2, 6):
        hip = h24p * 4 ** (-1.661 * log(h24p / h6p) / log(10)) * \
              6 ** (-1.285 * log(h6p / h1p) / log(10)) * \
              i ** (1.285 * log(h6p / h1p) / log(10))
        hour_hp.append(round(hip,2))    # round()函数用来对结果保留小数
    hour_hp.append(round(h6p,2))
    for i in range(7, 24):
        hip = h24p * 24 ** (-1.661 * log(h24p / h6p) / log(10)) * \
              i ** (1.661 * log(h24p / h6p) / log(10))
        hour_hp.append(round(hip,2))
    hour_hp.append(round(h24p,2))

    # 内插alpha值
    hour_alpha = []
    hour_hs = []
    for i in hour:
        row_hour = table.loc[i]
        alpha = np.interp(f, row_hour.index, row_hour.values)
        hs = hour_hp[i - 1] * alpha / 100
        hour_alpha.append(round(alpha,2))
        hour_hs.append(round(hs,2))

    hour_hs1 = [hour_hs[0]]
    for i in range(23):
        hs1 = hour_hs[i + 1] - hour_hs[i]
        hour_hs1.append(round(hs1,4))

    return hour_hp, hour_alpha, hour_hs, hour_hs1


def process_speculation(table, zone=1): # 一个区(zone)就是一个行，为了保留未来选区功能，此处默认为1，即第一行
    pailiexu = (table.iloc[zone]).to_list() # 抓取一行，输出为数组
    h1f = []   # 定义H1f为空数组用于接收成果
    for i in pailiexu:
        temp_list = hour_hs1[i-1] # 利用list固有的元素号来间接实现遍历排序，本来想的是把hour_hs1转化为dic类型,但发现list类型本身就有元素号
        h1f.append(round(temp_list,2))
    return h1f


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    p = '33.33'  # 计算频率P
    cvs = [0.7, 0.8, 0.9]  # cv1,cv6,cv24的值
    hs = [1, 10, 100]  # H1,H6,H24的值
    f = 350  # 面积
    hps = [calc_design_storm(kp, p, cvs[i], hs[i]) for i in range(len(cvs))]  # 设计暴雨计算
    hour_hp, hour_alpha, hour_hs, hour_hs1 = calc_hour_rainfall(ab, hps, f) # 各历时雨量计算
    h1f = process_speculation(sheet5)   # 暴雨过程推求
# See PyCharm help at https://www.jetbrains.com/help/pycharm/


