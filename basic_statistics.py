import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
# matplotlib.style.use('ggplot')

train = pd.read_csv('/Users/soup/Desktop/train.csv')
test = pd.read_csv('/Users/soup/Desktop/test.csv')

print(train.shape)
print(test.shape)

Y_label = train.Survived
train_no_survived = train.drop("Survived", axis=1)
x_train = train_no_survived.append(test)
print(x_train.shape)
print('------------')
print(x_train.columns.values)
print('------------')
print(x_train.head(4))
print('------------')
print(x_train.describe())
print('------------')
print(x_train.describe(include=['O']))
print('------------')
train_sex_nan = train.dropna(subset=["Sex"]) # 刪掉缺失Sex資料的
# plt.figure(num=1, figsize=(15, 8))
# plt.hist([train[train['Sex'] == 'male']['Age'], train[train['Sex'] == 'female']['Age']], stacked=False,
#         color=['b', 'r'], bins=30, label=['Male', 'Female'])
# plt.xlabel('Age')
# plt.ylabel('Number of Sex')
# plt.show()
Sex = train_sex_nan.groupby("Sex") # 將資料依據性別分組
print(Sex.size()) # 男生女生各有幾人
print('---------------------')
# print(Sex.get_group("male")) # 只顯示male的資料

survived_sex = train[train['Survived']==1]['Sex'].value_counts() # 從有生存的資料中計算各性別人數
print(survived_sex)
print('---------------------')
deaded_sex = train[train['Survived']==0]['Sex'].value_counts() # 從沒有生存的資料中計算各性別人數
print(deaded_sex)
print('---------------------')
df = pd.DataFrame([survived_sex, deaded_sex]) # 將有生存、沒生存的資料依照性別放成一表格
df.index = ['Survived', 'Dead'] # 將原本的tag改為存活、非存活
print(df)
print('---------------------')
# df.plot(kind='bar', stacked=False, color=['r', 'b']) # 依照性別畫倖存人數、非倖存人數
# plt.show() # 顯示圖
total_sex = train['Sex'].value_counts()

p_survived_sex = survived_sex / total_sex
print(p_survived_sex)
print('---------------------')
p_deaded_sex = deaded_sex / total_sex
print(p_deaded_sex)
print('---------------------')
p_df = pd.DataFrame([p_survived_sex, p_deaded_sex])
p_df.index = ['Survived', 'Dead']
print(p_df)
print('---------------------')
# p_df.plot(kind='bar', stacked=False, color=['r', 'b']) # 依照性別畫倖存人數比例、非倖存人數比例
# plt.show()
print(x_train['Name'])
print('---------------------')
survived_pclass = train[train['Survived']==1]['Pclass'].value_counts()
dead_pclass = train[train['Survived']==0]['Pclass'].value_counts()
pclass_df = pd.DataFrame([survived_pclass, dead_pclass])
pclass_df.index = ['Survived', 'Dead']
# pclass_df.plot(kind = 'bar', stacked=False)
# plt.show()
groupby_Pc = train[['Pclass', 'Survived', 'Sex']].groupby(['Pclass', 'Sex'], as_index=False).mean().sort_values(['Survived'],
                                                                                                  ascending=False)
# 不同性別與船票等級對於倖存率有無影響
print(groupby_Pc)
print('---------------------')

total_female_p1 = train[(train['Pclass']==1) & (train['Sex']=="female")]['Survived'].count() # 計算船等1&女性的倖存與非倖存者
diff_female_p1 = train[(train['Pclass']==1) & (train['Sex']=="female")]['Survived'].value_counts() # 分別計算船等1&女性的倖存、非倖存者
female_p1 = diff_female_p1/total_female_p1
total_female_p2 = train[(train['Pclass']==2) & (train['Sex']=="female")]['Survived'].count() # 計算船等2&女性的倖存與非倖存者
diff_female_p2 = train[(train['Pclass']==2) & (train['Sex']=="female")]['Survived'].value_counts() # 分別計算船等2&女性的倖存、非倖存者
female_p2 = diff_female_p2/total_female_p2
total_female_p3 = train[(train['Pclass']==3) & (train['Sex']=="female")]['Survived'].count() # 計算船等3&女性的倖存與非倖存者
diff_female_p3 = train[(train['Pclass']==3) & (train['Sex']=="female")]['Survived'].value_counts() # 分別計算船等3&女性的倖存、非倖存者
female_p3 = diff_female_p3/total_female_p3
female_df = pd.DataFrame([female_p1[[1, 0]], female_p2[[1, 0]], female_p3[[1, 0]]]) # [[1, 0]]為行的區分
female_df.index = ['Female in PC1', 'Female in PC2', 'Female in PC3']
# female_df.plot(kind='bar', stacked=False)
# plt.show()
print(female_df)
print('---------------------')

total_male_p1 = train[(train['Pclass']==1) & (train['Sex']=="male")]['Survived'].count() # 計算船等1&男性的倖存與非倖存者
diff_male_p1 = train[(train['Pclass']==1) & (train['Sex']=="male")]['Survived'].value_counts() # 分別計算船等1&男性的倖存、非倖存者
male_p1 = diff_male_p1/total_male_p1
total_male_p2 = train[(train['Pclass']==2) & (train['Sex']=="male")]['Survived'].count() # 計算船等2&男性的倖存與非倖存者
diff_male_p2 = train[(train['Pclass']==2) & (train['Sex']=="male")]['Survived'].value_counts() # 分別計算船等2&男性的倖存、非倖存者
male_p2 = diff_male_p2/total_male_p2
total_male_p3 = train[(train['Pclass']==3) & (train['Sex']=="male")]['Survived'].count() # 計算船等3&男性的倖存與非倖存者
diff_male_p3 = train[(train['Pclass']==3) & (train['Sex']=="male")]['Survived'].value_counts() # 分別計算船等3&男性的倖存、非倖存者
male_p3 = diff_male_p3/total_male_p3
male_df = pd.DataFrame([male_p1[[1, 0]], male_p2[[1, 0]], male_p3[[1, 0]]]) # [[1, 0]]為行的區分
male_df.index = ['Male in PC1', 'Male in PC2', 'Male in PC3']
# male_df.plot(kind='bar', stacked=False)
# plt.show()
print(male_df)
print('---------------------')
train_age_nan = train.dropna(subset=["Age"]) # 刪掉缺失Age資料的
# figure = plt.figure(figsize=(15,8))
# plt.hist([train_age_nan[train_age_nan['Survived']==1]['Age'], train_age_nan[train_age_nan['Survived']==0]['Age']], bins=30, stacked=True, color=['g','r'], label=['Survived', 'Dead'])
# 年齡與倖存關係
plt.xlabel('Age')
plt.ylabel('Number of passengers')
# plt.legend()
# plt.show()
train_fare_nan = train.dropna(subset=["Fare"]) # 刪掉缺失Fare資料的
# plt.hist([train_fare_nan[train_fare_nan['Survived']==1]['Fare'], train_fare_nan[train_fare_nan['Survived']==0]['Fare']], bins=30, stacked=True, color=['g','r'], label=['Survived', 'Dead'])
# 票價與倖存關係
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend(('Survived', 'Dead'))
# plt.show()
# 小結：得知票價越高 & 年齡越低 的倖存機率較高

ax = plt.subplot()
ax.scatter(train[train['Sex']=='female']['Age'], train[train['Sex']=='female']['Fare'], c='r', s=40)
ax.scatter(train[train['Sex']=='male']['Age'], train[train['Sex']=='male']['Fare'], c='g', s=40)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.legend(('Female', 'Male'))
# plt.show()
# x軸為年齡、y軸為票價畫散佈圖看有無相關性
# 小結：年齡、票價無關

train['Family'] = train['SibSp'] + train['Parch']
family_df = train[['Family', 'Survived']].groupby(['Family'], as_index=True).mean().sort_values(by='Survived', ascending=False)
print(family_df)
print('---------------------')
# .mean().sort_values(by='')是依照by輸入東西所出現的次數去計算平均
# 小結：有3位Family的存活率最大，依次為2位、1位

embarked_df = train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False)
print(embarked_df)
print('---------------------')
# 看上岸港口有無影響存活率
# 小結：在C港口上岸的竟然有0.55的存活率！

total_embarked_C = train[train['Embarked']=='C']['Survived'].count() # embarked為C的，無論是否survived都計數在一起
total_embarked_Q = train[train['Embarked']=='Q']['Survived'].count() # embarked為Q的，無論是否survived都計數在一起
total_embarked_S = train[train['Embarked']=='S']['Survived'].count() # embarked為S的，無論是否survived都計數在一起
survived_embarked_C = train[train['Embarked']=='C']['Survived'].value_counts() / total_embarked_C # embarked為C的，算有、無倖存率
survived_embarked_Q = train[train['Embarked']=='Q']['Survived'].value_counts() / total_embarked_Q # embarked為Q的，算有、無倖存率
survived_embarked_S = train[train['Embarked']=='S']['Survived'].value_counts() / total_embarked_S # embarked為S的，算有、無倖存率

p_embarked_df = pd.DataFrame([survived_embarked_C, survived_embarked_Q, survived_embarked_S])
p_embarked_df.index = ['Embarked_C', 'Embarked_Q', 'Embarked_S']
p_embarked_df.plot(kind='bar', stacked=False)
plt.legend(('Dead', 'Survived'))
plt.show()