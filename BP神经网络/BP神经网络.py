import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# 数据处理
df = pd.read_csv('泰坦尼克号数据.csv')
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('PassengerId', axis=1)
df = df.drop('Name', axis=1)
df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
df.Age = df.Age.fillna(df.Age.dropna().mean())
size_mapping = {"male": 1, "female": 0}
num_mapping = {"S": 1, "C": 0, "Q": 2}
df.Sex = df.Sex.map(size_mapping)
df.Embarked = df.Embarked.map(num_mapping)
ones = pd.DataFrame({'ones': np.ones(len(df))})
dt = df.copy().drop('Survived', axis=1)
x_train = np.matrix(pd.concat([ones, dt], axis=1)).T
y_train = np.matrix(df.Survived)


# 参数
n = int(np.power(x_train.shape[0] - 1, 0.5) + 1)
theta1 = np.random.randint(-1, 1, size=(n, x_train.shape[0])).astype(np.float64)
theta2 = np.random.randint(-1, 1, size=(1, n + 1)).astype(np.float64)
lamda = 5
alpha = 0.001


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(theta, X, y):
    return np.mean(np.multiply(-y, np.log(sigmoid(theta @ X))) - np.multiply((1 - y), np.log(1 - sigmoid(theta @ X))))


def r_cost(theta_1, theta_2):
    a = sigmoid(theta_1 @ x_train)
    a = np.insert(a, 0, 1, axis=0)
    r1 = np.power(theta_1[:, 1:], 2).sum()
    r2 = np.power(theta_2[:, 1:], 2).sum()
    r = (lamda / (2 * x_train.shape[1])) * (r1 + r2)
    c = cost(theta_2, a, y_train) + r
    return c


def gd(theta_1, theta_2, num):
    plt_x = []
    plt_y = []
    plt_x.append(0)
    plt_y.append(r_cost(theta_1, theta_2))
    for i in range(num):
        # 正向传播
        a = sigmoid(theta_1 @ x_train)
        a = np.insert(a, 0, 1, axis=0)
        h = sigmoid(theta_2 @ a)
        # 错误误差
        d1 = h - y_train
        d2 = np.multiply((theta_2.T @ d1), np.multiply(a, 1 - a))
        # 求正则化偏导
        D1 = np.zeros((n + 1, x_train.shape[0]))
        D2 = np.zeros((1, n + 1))
        D1 = D1 + d2 @ x_train.T
        D1 = D1[1:, :]
        D2 = D2 + d1 @ a.T
        PD1 = 1/x_train.shape[1] * D1
        theta_1[:, 0] = 0
        PD1 = PD1 + lamda * theta_1
        PD2 = 1/x_train.shape[1] * D2
        theta_2[:, 0] = 0
        PD2 = PD2 + lamda * theta_2
        theta_1 = theta_1 - alpha * PD1
        theta_2 = theta_2 - alpha * PD2
        plt_x.append(i+1)
        plt_y.append(r_cost(theta_1, theta_2))
    plt.plot(plt_x, plt_y)
    plt.show()
    return theta_1, theta_2


theta1, theta2 = gd(theta1, theta2, 5000)


def testing(dataset, labelset, theta_1, theta_2):
    # 记录预测正确的个数
    rightcount = 0
    a = sigmoid(theta_1 @ dataset)
    a = np.insert(a, 0, 1, axis=0)
    h = sigmoid(theta_2 @ a)
    for i in range(h.shape[1]):
        # 确定其预测标签
        if h[0, i] > 0.5:
            flag = 1
        else:
            flag = 0
        if labelset[0, i] == flag:
            rightcount += 1
        # 输出预测结果
        print("预测为%d   实际为%d" % (flag, labelset[0, i]))
    # 返回正确率
    return rightcount / x_train.shape[1]


s1 = testing(x_train, y_train, theta1, theta2)
print(s1)

print("--------------------------------------")


theta1 = np.random.randint(-1, 1, size=(n, x_train.shape[0])).astype(np.float64)
theta2 = np.random.randint(-1, 1, size=(1, n + 1)).astype(np.float64)


# 求偏导用一个数据
def sgd(theta_1, theta_2, num):
    plt_x = []
    plt_y = []
    plt_x.append(0)
    plt_y.append(r_cost(theta_1, theta_2))
    for i in range(num):
        m = np.random.randint(x_train.shape[1])
        # 正向传播
        a = sigmoid(theta_1 @ x_train[:, m])
        a = np.insert(a, 0, 1, axis=0)
        h = sigmoid(theta_2 @ a)
        # 错误误差
        d1 = h - y_train[:, m]
        d2 = np.multiply((theta_2.T @ d1), np.multiply(a, 1 - a))
        # 求正则化偏导
        D1 = np.zeros((n + 1, x_train[:, m].shape[0]))
        D2 = np.zeros((1, n + 1))
        D1 = D1 + d2 @ x_train[:, m].T
        D1 = D1[1:, :]
        D2 = D2 + d1 @ a.T
        PD1 = 1 / x_train[:, m].shape[1] * D1
        theta_1[:, 0] = 0
        PD1 = PD1 + lamda * theta_1
        PD2 = 1 / x_train[:, m].shape[1] * D2
        theta_2[:, 0] = 0
        PD2 = PD2 + lamda * theta_2
        theta_1 = theta_1 - alpha * PD1
        theta_2 = theta_2 - alpha * PD2
        plt_x.append(i+1)
        plt_y.append(r_cost(theta_1, theta_2))
    plt.plot(plt_x, plt_y)
    plt.show()
    return theta_1, theta_2


theta1, theta2 = sgd(theta1, theta2, 5000)

s2 = testing(x_train, y_train, theta1, theta2)
print(s2)

print("--------------------------------------")

theta1 = np.random.randint(-1, 1, size=(n, x_train.shape[0])).astype(np.float64)
theta2 = np.random.randint(-1, 1, size=(1, n + 1)).astype(np.float64)


def adam(theta_1, theta_2, num):
    plt_x = []
    plt_y = []
    plt_x.append(0)
    plt_y.append(r_cost(theta_1, theta_2))
    lamda_v = 0.9
    lamda_s = 0.999
    s1 = np.zeros(theta_1.shape)
    s2 = np.zeros(theta_2.shape)
    v1 = np.zeros(theta_1.shape)
    v2 = np.zeros(theta_2.shape)
    for i in range(num):
        m = np.random.randint(x_train.shape[1])
        # 正向传播
        a = sigmoid(theta_1 @ x_train[:, m])
        a = np.insert(a, 0, 1, axis=0)
        h = sigmoid(theta_2 @ a)
        # 错误误差
        d1 = h - y_train[:, m]
        d2 = np.multiply((theta_2.T @ d1), np.multiply(a, 1 - a))
        # 求正则化偏导
        D1 = np.zeros((n + 1, x_train[:, m].shape[0]))
        D2 = np.zeros((1, n + 1))
        D1 = D1 + d2 @ x_train[:, m].T
        D1 = D1[1:, :]
        D2 = D2 + d1 @ a.T
        PD1 = 1 / x_train[:, m].shape[1] * D1
        theta_1[:, 0] = 0
        PD1 = PD1 + lamda * theta_1
        PD2 = 1 / x_train[:, m].shape[1] * D2
        theta_2[:, 0] = 0
        PD2 = PD2 + lamda * theta_2
        v1 = lamda_v * v1 + (1-lamda_v) * PD1
        v2 = lamda_v * v2 + (1-lamda_v) * PD2
        vh1 = v1/(1-lamda_v**num)
        vh2 = v2/(1-lamda_v**num)
        s1 = lamda_s * s1 + (1-lamda_s) * np.power(PD1, 2)
        s2 = lamda_s * s2 + (1-lamda_s) * np.power(PD2, 2)
        sh1 = s1/(1-lamda_s**num)
        sh2 = s2/(1-lamda_s**num)
        P1 = (alpha * vh1)/(np.power(sh1, 0.5) + 1e-6)
        P2 = (alpha * vh2)/(np.power(sh2, 0.5) + 1e-6)
        theta_1 = theta_1 - P1
        theta_2 = theta_2 - P2
        plt_x.append(i+1)
        plt_y.append(r_cost(theta_1, theta_2))
    plt.plot(plt_x, plt_y)
    plt.show()
    return theta_1, theta_2


theta1, theta2 = adam(theta1, theta2, 5000)

s3 = testing(x_train, y_train, theta1, theta2)
print(s3)
