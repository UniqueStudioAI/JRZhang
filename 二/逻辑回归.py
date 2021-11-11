import pandas as pd
import numpy as np
import math


# 将'Pclass'部分数据删除
df = pd.read_csv('泰坦尼克号数据2.csv')
print(df.isnull().sum())


# 补零
df.Cabin = df.Cabin.fillna(0)


# 均值
df.Age = df.Age.fillna(df.Age.dropna().mean())


# 众数
df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values


# 归一化
def normalize(data):
    _min = data.min(0)  # 每一列最小值
    _max = data.max(0)
    ranges = _max - _min
    normal_matrix = data - np.tile(_min, (data.shape[0], 1))
    normal_matrix = normal_matrix / np.tile(ranges, (data.shape[0], 1))
    return normal_matrix


a = normalize(np.array(df.Age))
for f in range(len(a)):
    df.Age[f] = a[f][f]


# 独热编码
size_mapping = {"male": 1, "female": 0}
df.Sex = df.Sex.map(size_mapping)
df = df.join(pd.get_dummies(df.Embarked))
df = df.drop('Embarked', axis=1)


# 预处理数据
df = df.drop('PassengerId', axis=1)
df = df.drop('Survived', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('SibSp', axis=1)
df = df.drop('Parch', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('Fare', axis=1)
_data = np.array(df.dropna())
_labels = np.array(df.dropna().Pclass)
_pre_test = df[df.isnull().T.any()]
_index = list(_pre_test.index.values)
_test = np.array(_pre_test)


# 模型建立
def get_x(f):
    ones = pd.DataFrame({'ones': np.ones(len(f))})
    dt = pd.concat([ones, pd.DataFrame(f)], axis=1)
    return np.matrix(np.array(dt))


x_train = get_x(np.delete(_data, 0, axis=1))
P1 = np.array(_data[:, 0])
P2 = np.array(_data[:, 0])
P3 = np.array(_data[:, 0])
P1[P1 != 1.00000] = 0
P2[P2 != 2.00000] = 0
P2[P2 == 2.00000] = 1
P3[P3 != 3.00000] = 0
P3[P3 == 3.00000] = 1
y1_train = P1
y2_train = P2
y3_train = P3
_theta1 = np.zeros(len(x_train.T))
_theta2 = np.zeros(len(x_train.T))
_theta3 = np.zeros(len(x_train.T))
_alpha = 0.01
_iters = 1000
_lamda = 5


def hx(theta, x):
    return x @ theta.T


def s(z):
    return 1 / (1 + np.exp(-z))


# 正则化逻辑回归
def c(theta, x, y, lamda):
    _J = np.mean(-y * np.log(s(hx(theta, x)) + 1e-5) - (1 - y) * np.log(1 - s(hx(theta, x)) + 1e-5))
    m = x.shape[0]
    r = (lamda/2*m)*np.sum(np.power(theta[1:], 2))
    return _J + r


def g(x, y, theta, alpha, iters, lamda):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = len(x.T)
    cost = np.zeros(iters)
    for i in range(iters):
        gf = (s(hx(theta, x)) - y) @ x
        temp[0, 0] = np.matrix(theta)[0, 0] - (alpha * gf[0, 0])
        for m in range(1, parameters):
            temp[0, m] = np.matrix(theta)[0, m] - (alpha * gf[0, m]) - (lamda/x.shape[0])*temp[0, m]
        theta = temp
        cost[i] = c(theta, x, y, lamda)
    return theta, cost


_theta1, cost1 = g(x_train, y1_train, _theta1, _alpha, _iters, _lamda)
_theta2, cost2 = g(x_train, y2_train, _theta2, _alpha, _iters, _lamda)
_theta3, cost3 = g(x_train, y3_train, _theta3, _alpha, _iters, _lamda)

x_test = get_x(np.delete(_test, 0, axis=1))
a1 = s(hx(_theta1, x_test))
a2 = s(hx(_theta2, x_test))
a3 = s(hx(_theta3, x_test))
b1 = 1
b2 = 2
b3 = 3
p_answer = pd.read_csv('泰坦尼克号数据.csv')
answer = []
for a in range(len(_index)):
    value = p_answer.Pclass[_index[a]]
    answer = answer + [value]
correct = 0.0
for h in range(len(x_test)):
    if a1[h, 0] > a2[h, 0] and a1[h, 0] > a2[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b1, answer[h]))
        if b1 == answer[h]:
            correct += 1.0
    elif a2[h, 0] > a3[h, 0] and a2[h, 0] > a1[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b2, answer[h]))
        if b2 == answer[h]:
            correct += 1.0
    elif a3[h, 0] > a1[h, 0] and a3[h, 0] > a2[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b3, answer[h]))
        if b3 == answer[h]:
            correct += 1.0
print("正确率:{:.2f}%".format(correct / float(len(x_test)) * 100))


print("--------------------------------------")


# sgd优化每次迭代用一个样本(一行)
def sgd(x, y, alpha, iters):
    w = np.random.randn(x.shape[1])
    _w = np.empty(x.shape[1])
    _w = w
    m = len(x)
    d = len(x.T)
    pre_cost = 0
    for i in range(iters):
        for k in range(m):        # 先用一个样本
            for j in range(d):    # 把每个变量遍历一遍
                e = w[0]*x[k, 0]+w[1]*x[k, 1]+w[2]*x[k, 2]+w[3]*x[k, 3]+w[4]*x[k, 4]+w[5]*x[k, 5]-y[k]
                _w[j] = _w[j] - alpha*e*x[k, j]
            for j in range(d):
                w[j] = _w[j]
            cost = 0
            for k in range(m):
                cost += (w[0]*x[k, 0]+w[1]*x[k, 1]+w[2]*x[k, 2]+w[3]*x[k, 3]+w[4]*x[k, 4]+w[5]*x[k, 5]-y[k])**2
            cost = cost / m
            if abs(cost - pre_cost) < 0.001:
                break
            pre_cost = cost
        return w


__alpha = 0.01
__iters = 10000


w1 = sgd(x_train, y1_train, __alpha, __iters)
w2 = sgd(x_train, y2_train, __alpha, __iters)
w3 = sgd(x_train, y3_train, __alpha, __iters)


a11 = 1 / (1 + np.exp(- x_test @ w1.T))
a12 = 1 / (1 + np.exp(- x_test @ w2.T))
a13 = 1 / (1 + np.exp(- x_test @ w3.T))


_correct = 0.0
for h in range(len(x_test)):
    if a11[0, h] > a12[0, h] and a11[0, h] > a12[0, h]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b1, answer[h]))
        if b1 == answer[h]:
            _correct += 1.0
    elif a12[0, h] > a13[0, h] and a12[0, h] > a11[0, h]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b2, answer[h]))
        if b2 == answer[h]:
            _correct += 1.0
    elif a13[0, h] > a11[0, h] and a13[0, h] > a12[0, h]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b3, answer[h]))
        if b3 == answer[h]:
            _correct += 1.0
print("正确率:{:.2f}%".format(_correct / float(len(x_test)) * 100))


print("--------------------------------------")


# adam（运行时间较长...）
def adam(x, y):
    m, n = x.shape
    theta = np.zeros(n)
    alpha = 0.01
    threshold = 0.0001
    iterations = 1000
    b111 = 0.9         # 默认值
    b222 = 0.999       # 默认值
    e = 0.00000001     # 默认值
    mt = np.zeros(n)
    vt = np.zeros(n)
    for i in range(iterations):
        j = i % m
        error = 1 / (2 * m) * np.dot((s(hx(theta, x)) - y).T, (s(hx(theta, x)) - y))
        if abs(error.any()) <= threshold:
            break
        gf = (s(hx(theta, x)) - y) @ x
        gradient = gf[j, :]
        mt = b111 * mt + (1 - b111) * gradient
        vt = b222 * vt + (1 - b222) * (gradient.any()**2)
        mtt = mt / (1 - (b111**(i + 1)))
        vtt = vt / (1 - (b222**(i + 1)))
        vtt_sqrt = np.array([math.sqrt(vtt[0]),
                             math.sqrt(vtt[1]),
                             math.sqrt(vtt[2]),
                             math.sqrt(vtt[3]),
                             math.sqrt(vtt[4]),
                             math.sqrt(vtt[5])])
        theta = theta - alpha * mtt / (vtt_sqrt + e)
    return theta


w11 = adam(x_train, y1_train)
w22 = adam(x_train, y2_train)
w33 = adam(x_train, y3_train)


a111 = 1 / (1 + np.exp(- x_test @ w11.T))
a122 = 1 / (1 + np.exp(- x_test @ w22.T))
a133 = 1 / (1 + np.exp(- x_test @ w33.T))


__correct = 0.0
for h in range(len(x_test)):
    if a111[h, 0] > a122[h, 0] and a111[h, 0] > a122[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b1, answer[h]))
        if b1 == answer[h]:
            __correct += 1.0
    elif a122[h, 0] > a133[h, 0] and a122[h, 0] > a111[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b2, answer[h]))
        if b2 == answer[h]:
            __correct += 1.0
    elif a133[h, 0] > a111[h, 0] and a133[h, 0] > a122[h, 0]:
        print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (b3, answer[h]))
        if b3 == answer[h]:
            __correct += 1.0
print("正确率:{:.2f}%".format(__correct / float(len(x_test)) * 100))
