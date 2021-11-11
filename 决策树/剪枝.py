import pandas as pd
import numpy as np
from math import log
import operator
from 决策树 import t1
import drawtree
# 数据处理
df = pd.read_csv('泰坦尼克号数据2.csv')
df.Age = df.Age.fillna(df.Age.dropna().mean())
df.Embarked[df.Embarked.isnull()] = df.Embarked.dropna().mode().values
size_mapping = {"male": 1, "female": 0}
df.Sex = df.Sex.map(size_mapping)
df = df.drop('Name', axis=1)
df = df.drop('Ticket', axis=1)
df = df.drop('Cabin', axis=1)
df = df.drop('PassengerId', axis=1)
for i in range(len(df)):
    if df.Age[i] <= 18 and df.Age[i] > 0:
        df.Age[i] = 'Kid'
    elif df.Age[i] <= 60 and df.Age[i] > 18:
        df.Age[i] = 'Adult'
    elif df.Age[i] <= 80 and df.Age[i] > 60:
        df.Age[i] = 'Old'
for j in range(len(df)):
    if df.Fare[j] >= 0 and df.Fare[j] <=20:
        df.Fare[j] = 'Low'
    elif df.Fare[j] > 20 and df.Fare[j] <=100:
        df.Fare[j] = 'High'
    elif df.Fare[j] > 100:
        df.Fare[j] = 'Very High'
_data = pd.DataFrame(df.dropna())
_pre_test = df[df.isnull().T.any()]
_test = pd.DataFrame(_pre_test)
_index = list(_pre_test.index.values)
y_train = np.array(pd.DataFrame(_data, columns=['Pclass']))
x_train = np.array(pd.DataFrame(_data, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']))
x_test = np.array(pd.DataFrame(_test, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']))
x_data = np.array(pd.concat([pd.DataFrame(x_train), pd.DataFrame(y_train)], axis=1))
x_data1 = (x_data[0:700, :]).tolist()
x_data2 = (x_data[700: , :]).tolist()
x_labels = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']


def ent(data):
    num = len(data)
    label_counts = {}
    for featVec in data:
        current_label = featVec[-1]
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    s_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key])/num
        s_ent -= prob*log(prob, 2)
    return s_ent


def split1(data, axis, value):
    ret = []
    for featVec in data:
        if featVec[axis] == value:
            reduced_vec = featVec[:axis]
            reduced_vec.extend(featVec[axis+1:])
            ret.append(reduced_vec)
    return ret


def choose1(data):
    num_features = len(data[0]) - 1
    base_entropy = ent(data)
    best_gain = 0.0
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data = split1(data, i, value)
            prob = len(sub_data) / float(len(data))
            # 条件经验熵
            new_entropy += prob * ent(sub_data)
        gain = base_entropy - new_entropy
        if gain > best_gain:
            best_gain = gain
            best_feature = i
    return best_feature


def classify1(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]  # 获取树的第一特征属性
    second_dict = input_tree[first_str]  # 树的分子，子集合Dict
    feat_index = feat_labels.index(first_str)  # 获取决策树第一层在feat_labels中的位置
    class_label = 0
    for key in second_dict.keys():
        if feat_index != 7:
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':   # 迭代！
                    class_label = classify1(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
        else:
            class_label = key
    return class_label


# 按分类列表中的值，取出现次数最多的那个值（少数服从多数）
def majority_cnt(class_list):
    class_count = {}
    # 记录每个值出现的次数
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
            class_count[vote] += 1
        else:
            class_count[vote] += 1
        # 按字典中的value排序，降序
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]   # 输出现在得到的结果列中出现次数最多的值


# 计算在将feat此特征作为父节点情况下的错误率
def testing_feat(feat, train_data, test_data, labels):
    class_list = [example[-1] for example in train_data]  # 获取训练集的答案
    best_feat_index = labels.index(feat)  # 获取当前特征的索引数值
    train_data = [example[best_feat_index] for example in train_data]   # 获取训练集的此特征集
    test_data = [(example[best_feat_index], example[-1]) for example in test_data]  # 获取测试集的此特征集
    all_feat = set(train_data)
    error = 0.0
    for value in all_feat:    # 对特征集的每个结果迭代
        class_feat = [class_list[i] for i in range(len(class_list)) if train_data[i] == value]
        major = majority_cnt(class_feat)
        for data in test_data:
            if data[0] == value and data[1] != major:
                error += 1.0
    return error


# 计算此数对测试集的泛化能力（错误率）
def testing(my_tree, data_test, labels):
    error = 0.0
    for i in range(len(data_test)):
        if classify1(my_tree, labels, data_test[i]) != data_test[i][-1]:
            error += 1
    return float(error)


# 计算若将major（出现次数最多的结果值）作为子节点的错误率
def testing_major(major, data_test):
    error = 0.0
    for i in range(len(data_test)):
        if major != data_test[i][-1]:
            error += 1
    return float(error)


def create_tree(dataSet, labels, data_full, labels_full, test_data, mode):
    class_list = [example[-1] for example in dataSet]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)
    labels_copy = labels.copy()
    best_feat = choose1(dataSet)
    best_feat_label = labels[best_feat]
    if mode == "post":
        my_tree = {best_feat_label: {}}
    elif mode == "prev":
        if testing_feat(best_feat_label, dataSet, test_data, labels_copy) < testing_major(majority_cnt(class_list),
                                                                                          test_data):
            my_tree = {best_feat_label: {}}
        else:
            return majority_cnt(class_list)
    feat_values = [example[best_feat] for example in dataSet]
    unique_vals = set(feat_values)
    del (labels[best_feat])
    for value in unique_vals:
        sub_labels = labels[:]
        # 迭代！
        my_tree[best_feat_label][value] = create_tree(split1(dataSet, best_feat, value), sub_labels, data_full,
                                                      labels_full, split1(test_data, best_feat, value), mode)
    if mode == "post":
        # 比较   当前树错误率  和   以多数投票为节点的错误率
        if testing(my_tree, test_data, labels_copy) > testing_major(majority_cnt(class_list), test_data):
            return majority_cnt(class_list)
    return my_tree


train_data, test_data, labels = x_data1, x_data2, x_labels
data_full = train_data[:]
labels_full = labels[:]
mode1 = "prev"
mode2 = "post"
t2 = create_tree(train_data, labels, data_full, labels_full, test_data, mode1)
t3 = create_tree(train_data, labels, data_full, labels_full, test_data, mode2)
drawtree.createPlot(t1)
drawtree.createPlot(t3)
drawtree.createPlot(t2)
