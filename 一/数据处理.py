import pandas as pd
import numpy as np
import operator


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
for j in range(len(a)):
    df.Age[j] = a[j][j]


# 独热编码
size_mapping = {"male": 1, "female": 0}
df.Sex = df.Sex.map(size_mapping)
df = df.join(pd.get_dummies(df.Embarked))
df = df.drop('Embarked', axis=1)


# knn
def knn(test, data, labels, k):
    distance = (np.tile(test, (data.shape[0], 1)) - data) ** 2
    ad_distance = distance.sum(axis=1)
    sq_distance = ad_distance ** 0.5
    ed_distance = sq_distance.argsort()  # 将sq_distance中的元素从小到大排列，提取其对应的index(索引数值)，然后输出到ed_distance
    _dict = {}
    for i in range(k):
        vote_label = labels[ed_distance[i]]  # 选择k个最近邻标签
        # 将标签出现的次数储存到_dict中，_dict中没有对应标签，则_dict.get(vote_label, 0)返回0
        _dict[vote_label] = _dict.get(vote_label, 0) + 1
        # items()方法的遍历：items()把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
        # operator.itemgetter(1)按照第二个元素的次序对元组进行排序
        # key 主要是用来进行比较的元素，只有一个参数，具体的函数的参数就是取自于可迭代对象中，指定可迭代对象中的一个元素来进行排序。
        # reverse = True 降序 ， reverse = False 升序（默认）
    sort_dict = sorted(_dict.items(), key=operator.itemgetter(1), reverse=True)
    # 返回出现次数最多的标签
    return sort_dict[0][0]


# 预处理数据
df = df.drop('PassengerId', axis=1)
df = df.drop('Survived', axis=1)
df = df.drop('Name', axis=1)
df = df.drop('SibSp', axis=1)
df = df.drop('Parch', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('Fare', axis=1)
_data = np.array(df.dropna().drop('Pclass', axis=1))
_labels = np.array(df.dropna().Pclass)
_pre_test = df[df.isnull().T.any()]
_index = list(_pre_test.index.values)
_test = np.array(_pre_test.drop('Pclass', axis=1))



p_answer = pd.read_csv('泰坦尼克号数据.csv')
answer = []
for a in range(len(_index)):
    value = p_answer.Pclass[_index[a]]
    answer = answer + [value]
n = _test.shape[0]
correct = 0.0
for m in range(n):
    result = knn(_test[m, :], _data, _labels, 15)
    print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (result, answer[m]))
    if result == answer[m]:
        correct += 1.0
print("正确率:{:.2f}%".format(correct / float(n) * 100))
