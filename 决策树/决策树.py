import pandas as pd
import numpy as np
from math import log
import operator


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
y_train = pd.DataFrame(_data, columns=['Pclass'])
x_train = pd.DataFrame(_data, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
x_test = np.array(pd.DataFrame(_test, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']))
x_data = np.array(pd.concat([x_train, y_train], axis=1)).tolist()
x_labels = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']


# 经验熵(算出各种类标签出现的频率作为概率代入公式)
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


# 划分数据集(根据某个特征是不是某个值)
def split1(data, axis, value):
    ret = []
    for featVec in data:
        if featVec[axis] == value:
            reduced_vec = featVec[:axis]
            reduced_vec.extend(featVec[axis+1:])
            ret.append(reduced_vec)
    return ret


# ID3选信息增益最大的
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


# 统计出现最多的类标签
def majority_cnt(class_list):
    class_count = {}
    # 统计classList中每个元素出现的次数
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
            class_count[vote] += 1
        # 根据字典的值降序排列
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_class_count[0][0]


featLabels = []


# 决策树
def create_tree(data, labels, feat_labels, choose=choose1):
    # 取分类标签（是否放贷：yes or no）
    class_list = [example[-1] for example in data]
    # 如果类别完全相同则停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(data[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征
    best_feat = choose(data)
    # 最优特征的标签
    best_feat_label = labels[best_feat]
    feat_labels.append(best_feat_label)
    # 根据最优特征的标签生成树
    my_tree = {best_feat_label: {}}
    # 删除已经使用的特征标签
    # 得到训练集中所有最优解特征的属性值
    feat_values = [example[best_feat] for example in data]
    # 去掉重复的属性值
    unique_vals = set(feat_values)
    # 遍历特征，创建决策树
    for value in unique_vals:
        del_best = best_feat
        del_labels = labels[best_feat]
        del (labels[best_feat])
        my_tree[best_feat_label][value] = create_tree(split1(data, best_feat, value), labels, feat_labels)
        labels.insert(del_best, del_labels)
    return my_tree


# C4.5选信息增益率最大的
def choose2(data):
    num_feature = len(data[0]) - 1
    base_entropy = ent(data)
    best_info_gain_ratio = 0.0
    best_feature_idx = -1
    for feature_idx in range(num_feature):
        feature_val_list = [number[feature_idx] for number in data]
        unique_feature_val_list = set(feature_val_list)
        new_entropy = 0
        split_info = 0.0
        for value in unique_feature_val_list:
            sub_data_set = split1(data, feature_idx, value)
            prob = len(sub_data_set) / float(len(data))
            new_entropy += prob * ent(sub_data_set)
            split_info += -prob * log(prob, 2)
        info_gain = base_entropy - new_entropy
        # fix the overflow bug
        if split_info == 0:
            continue
        # 信息增益率=信息增益/此特征的经验熵
        info_gain_ratio = info_gain / split_info
        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain_ratio
            best_feature_idx = feature_idx
    return best_feature_idx


t1 = create_tree(x_data, x_labels, featLabels, choose=choose1)
t2 = create_tree(x_data, x_labels, featLabels, choose=choose2)


def classify1(input_tree, feat_labels, test_vec):
    first_str = list(input_tree.keys())[0]  # 获取树的第一特征属性
    second_dict = input_tree[first_str]  # 树的分子，子集合Dict
    feat_index = feat_labels.index(first_str)  # 获取决策树第一层在feat_labels中的位置
    class_label = 0
    for key in second_dict.keys():
        if feat_index != 7:
            if test_vec[feat_index] == key:
                if type(second_dict[key]).__name__ == 'dict':
                    class_label = classify1(second_dict[key], feat_labels, test_vec)
                else:
                    class_label = second_dict[key]
        else:
            class_label = key
    return class_label


answer1 = []
for i in range(len(x_test.tolist())):
    ai = classify1(t1, x_labels, x_test.tolist()[i])
    answer1.append(ai)

answer2 = []
for i in range(len(x_test.tolist())):
    aii = classify1(t2, x_labels, x_test.tolist()[i])
    answer2.append(aii)


p_answer = pd.read_csv('泰坦尼克号数据.csv')
answer = []
for a in range(len(_index)):
    value = p_answer.Pclass[_index[a]]
    answer = answer + [value]


correct = 0.0
for h in range(len(x_test)):
    print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (answer1[h], answer[h]))
    if answer1[h] == answer[h]:
        correct += 1.0
print("正确率:{:.2f}%".format(correct / float(len(x_test)) * 100))

print("--------------------------------------")

_correct = 0.0
for h in range(len(x_test)):
    print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (answer2[h], answer[h]))
    if answer2[h] == answer[h]:
        _correct += 1.0
print("正确率:{:.2f}%".format(_correct / float(len(x_test)) * 100))

print("--------------------------------------")


# CART
for i in range(len(df)):
    if df.Age[i] == 'Adult':
        df.Age[i] = 0
    else:
        df.Age[i] = 1
    if df.SibSp[i] > 0:
        df.SibSp[i] = 1
    else:
        df.SibSp[i] = 0
    if df.Parch[i] > 0:
        df.Parch[i] = 1
    else:
        df.Parch[i] = 0
    if df.Fare[i] == 'Low':
        df.Fare[i] = 0
    else:
        df.Fare[i] = 1
    if df.Embarked[i] == 'S':
        df.Embarked[i] = 0
    else:
        df.Embarked[i] = 1
__data = pd.DataFrame(df.dropna())
__pre_test = df[df.isnull().T.any()]
__test = pd.DataFrame(__pre_test)
__index = list(__pre_test.index.values)
_y_train = pd.DataFrame(__data, columns=['Pclass'])
_x_train = pd.DataFrame(__data, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])
_x_test = np.array(pd.DataFrame(__test, columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']))
_x_data = np.array(pd.concat([_x_train, _y_train], axis=1)).tolist()
_x_labels = ['Survived', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Pclass']


def pb(data):
    num = len(data)
    fea_counts = 0
    fea1 = data[0][len(data[0])-1]
    for fea_vec in data:
        if fea_vec[-1] == fea1:
            fea_counts += 1
    prob_ent = float(fea_counts) / num
    return prob_ent


def choose3(data):
    num_features = len(data[0]) - 1
    if num_features == 1:
        return 0
    best_gini = 1
    best_feature = -1
    for i in range(num_features):
        feat_list = [example[i] for example in data]
        unique_vals = set(feat_list)
        new_entropy = 0.0
        for value in unique_vals:
            sub_data = split1(data, i, value)
            prob = len(sub_data) / float(len(data))
            new_entropy += prob * 2 * pb(data) * (1-pb(data))
            if new_entropy < best_gini:
                best_gini = new_entropy
                best_feature = i
        return best_feature


t3 = create_tree(_x_data, _x_labels, featLabels, choose=choose3)

answer3 = []
for i in range(len(_x_test.tolist())):
    aiii = classify1(t3, _x_labels, _x_test.tolist()[i])
    answer3.append(aiii)

__correct = 0.0
for h in range(len(_x_test)):
    print('预测此人客舱等级为：%d\t此人真实船舱等级为：%d' % (answer3[h], answer[h]))
    if answer3[h] == answer[h]:
        __correct += 1.0
print("正确率:{:.2f}%".format(__correct / float(len(_x_test)) * 100))
