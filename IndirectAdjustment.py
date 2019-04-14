# -*- coding: utf-8 -*-

import numpy as np

# 基线数据
line = [[0 for i in range(6)] for j in range(24)]
line = np.loadtxt("source.txt")

# 点位数据，将P1(0, 0, 0)置为已知点
point = [[0 for i in range(3)] for j in range(14)]
point[0] = [0, 0, 0]
for i in range(1, 14, 1):
    for j in range(0, 24, 1):
        if line[j][2] == i + 1 and line[j][1] <= i:
            point[i] = [
                x + y for x, y in zip(point[int(line[j][1] - 1)], line[j][3:])
            ]
        elif line[j][1] == i + 1 and line[j][2] <= i:
            point[i] = [
                x - y for x, y in zip(point[int(line[j][2] - 1)], line[j][3:])
            ]

# 从基线数据中提取系数构造B矩阵，将基线中起点位置对应的系数置为-1，终点位置对应的系数置为1
matrix_B = [[0 for i in range(14)] for j in range(24)]
for i in range(0, 24):
    matrix_B[i][int(line[i][2]) - 1] = 1
    matrix_B[i][int(line[i][1]) - 1] = -1

# 提取基线数据中的x, y, z增量，方便后续计算
line_data = [line[iter][3:6] for iter in range(0, 24)]

# 计算误差方程常数项
l = line_data - np.dot(np.array(matrix_B), np.array(point))

# 等权观测，构造单位对角阵
matrix_P = np.eye(24)

# 计算法方程的解x
Nbb = np.dot(np.dot(np.transpose(matrix_B), matrix_P), matrix_B)
W = np.dot(np.dot(np.transpose(matrix_B), matrix_P), l)
x = np.dot(np.linalg.inv(Nbb), W)

# 将x带入误差方程，求得改正数
V = np.dot(matrix_B, x) - l

# 由基线观测值与改正数算得平差值
result = line_data + V

# 输出改正数矩阵
np.savetxt("Correction.txt", V)

# 输出平差值
np.savetxt("AdjustedValue.txt", result)

# 核验
#print(result[7][:]+result[8][:]+result[9][:]+result[16][:])
