import random
import pandas as pd
import numpy as np


# 将'Pclass'部分数据删除
df = pd.read_csv('泰坦尼克号数据2.csv')


# 独热编码
df = df.join(pd.get_dummies(df.Embarked))
df = df.drop('Embarked', axis=1)


# 数据处理
df = df.drop('PassengerId', axis=1)
df = df.drop('Survived', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('SibSp', axis=1)
df = df.drop('Parch', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('Sex', axis=1)
df = df.drop('Age', axis=1)
_pre_test = df[df.isnull().T.any()]
_test = np.array(_pre_test)[:, 1:]
_index = list(_pre_test.index.values)


# K-Means
def distance(x1, x2):
    return np.sqrt(sum(np.power(x1 - x2, 2)))


def inp(data, k):
    n = data.shape[1]
    points = np.matrix(np.zeros((k, n)))
    for j in range(n):
        _min = min(data[:, j])
        _range = float(max(data[:, j]) - _min)
        points[:, j] = np.matrix(_min + _range * np.random.rand(k, 1))
    return points


def km(data, k, f=distance, create_cent=inp):
    m = data.shape[0]
    cluster_assessment = np.matrix(np.zeros((m, 2)))
    centroids = create_cent(data, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        for i in range(m):
            min_dist = np.inf
            min_index = -1
            for j in range(k):
                dist_ji = f(centroids[j, :], data[i, :])
                for n in range(len(dist_ji.T)):
                    if dist_ji[0, n] < min_dist:
                        min_dist = dist_ji[0, n]
                        min_index = j
            if cluster_assessment[i, 0] != min_index:
                cluster_changed = True
            cluster_assessment[i, :] = min_index, min_dist**2
        print(centroids)
        for cent in range(k):
            # 矩阵.A变为ndarray类型；（）则过滤，只索引簇序号对应的数据；np.nonzero返回数组中非零元素的索引值数组。
            pre_centroids = data[np.nonzero(cluster_assessment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(pre_centroids, axis=0)
    return centroids, cluster_assessment


a, b = km(_test, 3)


# 这里其实只用了票价的聚类结果.........
p_answer = pd.read_csv('泰坦尼克号数据.csv')
b1 = 1
b2 = 2
b3 = 3
answer = []
for a in range(len(_index)):
    value = p_answer.Pclass[_index[a]]
    answer = answer + [value]
correct = 0.0
for i in range(len(b)):
    if b[i, 0] == 0:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b3, answer[i]))
        if b3 == answer[i]:
            correct += 1.0
    if b[i, 0] == 1:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b1, answer[i]))
        if b1 == answer[i]:
            correct += 1.0
    if b[i, 0] == 2:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b2, answer[i]))
        if b2 == answer[i]:
            correct += 1.0
print("正确率:{:.2f}%".format(correct / float(len(b)) * 100))


print("--------------------------------------")


# K-Means++（多运行几次正确率可高达97%）
def nearest(point, center):
    min_dist = np.inf
    m = center.shape[1]
    for i in range(m):
        d = distance(point, center[:, i])
        for j in range(len(d.T)):
            if min_dist > d[0, j]:
                min_dist = d[0, j]
    return min_dist


def _inp(data, k):
    m, n = data.shape
    cluster_centers = np.matrix(np.zeros((k, n)))
    # 随机抽取一个样本作为第一个中心
    index = np.random.randint(0, m)
    cluster_centers[0, :] = np.copy(data[index, :])
    d = [0.0 for _ in range(m)]
    for h in range(1, k):
        sum_all = 0
        for j in range(m):
            # 对每一个样本找到最近的聚类中心点
            d[j] = nearest(data[j, :], cluster_centers)
            sum_all += d[j]
        # random.random()生成（0，1）随机数
        sum_all *= random.random()
        # 获得距离最远的样本点作为聚类中心点，enumerate()遍历每个元素输出（索引，值）数组
        for p, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_centers[h, :] = np.copy(data[p, :])
            break
    return cluster_centers


_a, _b = km(_test, 3, create_cent=_inp)


for i in range(len(b)):
    if b[i, 0] == 0:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b3, answer[i]))
        if b3 == answer[i]:
            correct += 1.0
    if b[i, 0] == 1:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b1, answer[i]))
        if b1 == answer[i]:
            correct += 1.0
    if b[i, 0] == 2:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b2, answer[i]))
        if b2 == answer[i]:
            correct += 1.0
print("正确率:{:.2f}%".format(correct / float(len(b)) * 100))
