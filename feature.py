import pandas as pd
import random as rd
train = pd.read_csv('/Users/soup/Desktop/train.csv')
test = pd.read_csv('/Users/soup/Desktop/test.csv')
Y_label = train['Survived']
print(test)
All_titanic = train.append(test) # train, test合併資料
All_titanic['Family'] = All_titanic['SibSp'] + All_titanic['Parch'] # 將所有親人人數加總成family人數

All_titanic.drop('SibSp',1,inplace=True)
All_titanic.drop('Parch',1,inplace=True)
print('-------------------')
print(All_titanic[All_titanic['Age'].isnull()]) # 印Age有缺失的資料出來
# print(All_titanic.columns.values) # 印各行的title
print('-------------------')
for title in['Mr\.','Major\.','Sir\.','Master\.','Dr\.']:
    num = train[(train['Name'].str.contains(title))]['Name'].count() # for loop計數以上4個名字
    age_mean = round(train[train['Name'].str.contains(title)]['Age'].mean(), 1) # for loop計算以上4個名字的平均年齡
    age_median = round(train[train['Name'].str.contains(title)]['Age'].median(), 1)  # for loop計算以上4個名字的中位數年齡
    num_survived = train[train['Survived'] == 1 & train['Name'].str.contains(title)]['Name'].count() # for loop計算以上4個名字的倖存人數
    num_died = train[train['Survived']== 0 & train['Name'].str.contains(title)]['Name'].count() # for loop計算以上4個名字的死亡人數
    num_total = num_survived+num_died
    print(num)
    print(age_mean)
    print(age_median)
    print(num_survived)
    print(num_died)
    print(num_total)
    print('------------------------')

for title in['Ms\.','Miss\.','Mrs\.','Lady\.']:
    num = train[(train['Name'].str.contains(title))]['Name'].count() # for loop計數以上4個名字
    age_mean = round(train[train['Name'].str.contains(title)]['Age'].mean(), 1) # for loop計算以上4個名字的平均年齡
    age_median = round(train[train['Name'].str.contains(title)]['Age'].median(), 1)  # for loop計算以上4個名字的中位數年齡
    num_survived = train[train['Survived'] == 1 & train['Name'].str.contains(title)]['Name'].count() # for loop計算以上4個名字的倖存人數
    num_died = train[train['Survived']== 0 & train['Name'].str.contains(title)]['Name'].count() # for loop計算以上4個名字的死亡人數
    num_total = num_survived+num_died
    print(num)
    print(age_mean)
    print(age_median)
    print(num_survived)
    print(num_died)
    print(num_total)
    print('------------------------')
# 將Age資料有缺失的補完
def fill_Age():
    global All_titanic
    mask = (All_titanic['Age'].isnull()) & (
                All_titanic['Name'].str.contains('Miss') | All_titanic['Name'].str.contains('Ms.')
                | All_titanic['Name'].str.contains('Mrs.'))  # 遺失Age資料以及名字有Miss, Ms.或Mrs.的
    mask2 = (All_titanic['Name'].str.contains('Miss') | All_titanic['Name'].str.contains('Ms.') | All_titanic[
        'Name'].str.contains('Mrs.'))  # 名字有Miss, Ms.或Mrs.的資料
    All_titanic.loc[mask, 'Age'] = All_titanic.loc[mask, 'Age'].fillna(All_titanic.loc[mask2, 'Age'].median())
    # .loc為標籤，.loc[mask, 'Age']意思是符合mask條件的資料中的Age那一行

    mask = (All_titanic['Age'].isnull()) & (
                All_titanic['Name'].str.contains('Mr.') | All_titanic['Name'].str.contains('Major')
                | All_titanic['Name'].str.contains('Sir'))  # 遺失Age資料以及名字有Mr., Major.或Sir的
    mask2 = (All_titanic['Name'].str.contains('Mr.') | All_titanic['Name'].str.contains('Major') | All_titanic[
        'Name'].str.contains('Sir'))  # 名字有Mr., Major或Sir的資料
    All_titanic.loc[mask, 'Age'] = All_titanic.loc[mask, 'Age'].fillna(All_titanic.loc[mask2, 'Age'].median())
    # .loc為標籤，.loc[mask, 'Age']意思是符合mask條件的資料中的Age那一行

    mask = (All_titanic['Age'].isnull()) & (
                All_titanic['Name'].str.contains('Master') | All_titanic['Name'].str.contains('Dr.'))
    mask2 = (All_titanic['Name'].str.contains('Master') | All_titanic['Name'].str.contains('Dr.'))
    All_titanic.loc[mask, 'Age'] = All_titanic.loc[mask, 'Age'].fillna(All_titanic.loc[mask2, 'Age'].median())
    # 若mask後面有“或”，那麼在.loc[].fillna()時會依照符合的條件填進去
fill_Age()

# 將Fare資料有缺失的補完
def fill_Fare():
    All_titanic['Fare'].fillna(All_titanic['Fare'].median(), inplace=True) # inplace=False會顯示Fare缺失值
fill_Fare()



# 將Embarked資料有缺失的補完
def fill_Embarked():
    All_titanic['Embarked'].fillna("C", inplace=True)
fill_Embarked()
print(All_titanic.info())
print(All_titanic.describe())
print(All_titanic.describe(include=['O'])) # include=['O']為Object欄位的意思
print('--------------------------------')
Titanic_cabin_nan = All_titanic[All_titanic['Cabin'].isnull()] # 缺失cabin資料的
Titanic_cabin_notnan = All_titanic[All_titanic['Cabin'].notnull()] # "沒有"缺失cabin資料的

# 現在Cabin缺失1000多筆資料、只有263筆資料，要想辦法補足，所以想辦法看看他與其他的關係
print(pd.value_counts(Titanic_cabin_nan['Survived'])) # 查看遺失Cabin資料的死亡、倖存人數
print(pd.value_counts(Titanic_cabin_notnan['Survived'])) # 查看沒有遺失Cabin資料的死亡、倖存人數
print(pd.value_counts(Titanic_cabin_nan['Pclass'])) # 查看遺失Cabin資料的Pclass等級個別人數
print(pd.value_counts(Titanic_cabin_notnan['Pclass'])) # 查看沒有遺失Cabin資料的Pclass等級個別人數
# 小結：遺失Cabin資料的死亡率高、沒有遺失Cabin資料的倖存率高、遺失Cabin資料的Pclass等級低、沒有遺失Cabin資料的Pclass等級高
# 所以死亡率高連結到Pclass等級低：survived==0的Pclass=1、所以倖存率高連結到Pclass等級高：survived==1的Pclass=3
# 沒有survievd資料的

print(Titanic_cabin_nan.columns.values) # 查看行向量的value(index)
print(Titanic_cabin_notnan[['Pclass', 'Cabin']]) # 只看其中的某兩行

# 我想看沒有All_titanic資料中，Pclass=1的Cabin number主要是哪一類
print('A:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('A'))]['Pclass'].count())
print('B:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('B'))]['Pclass'].count())
print('C:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('C'))]['Pclass'].count())
print('D:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('D'))]['Pclass'].count())
print('E:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('E'))]['Pclass'].count())
print('F:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('F'))]['Pclass'].count())
print('G:')
print(All_titanic[(All_titanic['Pclass']==1) & (All_titanic['Cabin'].str.contains('G'))]['Pclass'].count())
print('----------------------')
print('A:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('A'))]['Pclass'].count())
print('B:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('B'))]['Pclass'].count())
print('C:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('C'))]['Pclass'].count())
print('D:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('D'))]['Pclass'].count())
print('E:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('E'))]['Pclass'].count())
print('F:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('F'))]['Pclass'].count())
print('G:')
print(All_titanic[(All_titanic['Pclass']==2) & (All_titanic['Cabin'].str.contains('G'))]['Pclass'].count())
print('----------------------')
print('A:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('A'))]['Pclass'].count())
print('B:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('B'))]['Pclass'].count())
print('C:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('C'))]['Pclass'].count())
print('D:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('D'))]['Pclass'].count())
print('E:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('E'))]['Pclass'].count())
print('F:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('F'))]['Pclass'].count())
print('G:')
print(All_titanic[(All_titanic['Pclass']==3) & (All_titanic['Cabin'].str.contains('G'))]['Pclass'].count())
print('----------------------')
# 小結：主要Pclass=1的Cabin number=C、主要Pclass=2的Cabin number=F、主要Pclass=3的Cabin number=G

def fill_Cabin():
    mask = (All_titanic['Cabin'].isnull()) & (All_titanic['Survived']==0)
    # 遺失Cabin資料以及Survived=0的
    All_titanic.loc[mask, 'Cabin'] = All_titanic.loc[mask, 'Cabin'].fillna('C')
    # .loc為標籤，.loc[mask, 'Cabin']意思是符合mask條件的資料中的Cabin那一行
    mask = (All_titanic['Cabin'].isnull()) & (All_titanic['Survived'] == 1)
    All_titanic.loc[mask, 'Cabin'] = All_titanic.loc[mask, 'Cabin'].fillna('G')
    mask = (All_titanic['Survived'].isnull()) & (All_titanic['Pclass']==1)
    All_titanic.loc[mask,'Cabin'] = All_titanic.loc[mask, 'Cabin'].fillna('C')
    # 缺失Survived資料的 & Pclass=1的Cabin填C
    mask = (All_titanic['Survived'].isnull()) & (All_titanic['Pclass']==2)
    All_titanic.loc[mask,'Cabin'] = All_titanic.loc[mask, 'Cabin'].fillna('F')
    # 缺失Survived資料的 & Pclass=2的Cabin填F
    mask = (All_titanic['Survived'].isnull()) & (All_titanic['Pclass']==3)
    All_titanic.loc[mask,'Cabin'] = All_titanic.loc[mask, 'Cabin'].fillna('G')
    # 缺失Survived資料的 & Pclass=3的Cabin填G
fill_Cabin()
print(All_titanic.info())
# print(All_titanic)

# 將Ticket欄位只取字串、數字省略掉，這樣可以依字串來進行分類，原本加上數字有9XX種，現在只剩21種
def fill_Ticket():
    All_titanic['Ticket'] = All_titanic['Ticket'].str.extract('([a-zA-Z]*)', expand=False).str.upper()
    # str.extract('([a-zA-Z]*)')為僅取字串部分
fill_Ticket()
print(All_titanic.describe(include=['O']))
print(All_titanic['Ticket'].unique()) # 看unique有哪幾個value
def onehot_encoding():
    # All_titanic.drop("PassengerId", inplace=True, axis=1)
    # 刪除PassengerId那行，inplace=True為將原本的檔案那行刪，inplace=False的話，要指定到另一個新的檔名
    Sex_mapping = {'male': 0, 'female': 1}
    All_titanic['Sex'] = All_titanic['Sex'].map(Sex_mapping)
    # 將性別用.map() 轉成男：0、女：1
    All_titanic['Embarked'] = All_titanic['Embarked'].astype('category').cat.codes
    All_titanic['Pclass'] = All_titanic['Pclass'].astype('category').cat.codes
    All_titanic['Cabin'] = All_titanic['Cabin'].astype('category').cat.codes
    All_titanic['Ticket'] = All_titanic['Ticket'].astype('category').cat.codes
    # .astype('category').cat.codes為一種資料轉數值的寫法，要查！！！！！！！！！！！！！！
onehot_encoding()

# 開始建構ML模型
All_titanic.drop('Name', inplace=True, axis=1) # 將名字欄位刪掉
Y_train = All_titanic.head(891).Survived
All_titanic = All_titanic.drop('Survived', axis=1)
test_titanic = All_titanic.iloc[891:]
train_titanic = All_titanic.head(891)

data_test = train_titanic[['Age', 'Pclass', 'Sex', 'Embarked', 'Family']]

print(test_titanic)
print('--------------')
print(train_titanic)
print('--------------')
# ML開始
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn import metrics, cross_validation

print('-----------')
lin_clf = LinearSVC()
lin_clf.fit(train_titanic, Y_train.values.ravel())
ttt = lin_clf.predict(test_titanic)
# ggg = [rd.randint(0,1)for _ in range(418)]
print(ttt)
submission = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": ttt})
submission.to_csv('Titanic-submission.csv', index=False)


