import pandas as pd
import numpy as np
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




