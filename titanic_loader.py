
'''
PassengerId：乘客序号；

Survived：最终是否存活（1表示存活，0表示未存活）；

Pclass：舱位，1是头等舱，3是最低等；

Name：乘客姓名；

Sex：性别；

Age：年龄；

SibSp：一同上船的兄弟姐妹或配偶；

Parch：一同上船的父母或子女；

Ticket：船票信息；

Fare：乘客票价，决定了Pclass的等级；

Cabin：客舱编号，不同的编号对应不同的位置；

Embarked：上船地点，主要是S（南安普顿）、C（瑟堡）、Q（皇后镇）。

'''

#### Libraries
import pandas as pd 
import numpy as np
import sklearn.preprocessing as preprocessing
import csv  
 

def load_data(s):# csv数据读取
    if s == 0:
        df = pd.read_csv('Titanic/data/train.csv') #读取训练数据
    else:
        df = pd.read_csv('Titanic/data/test.csv') #读取竞赛数据

    #将类别变量转换为数值
    df = set_Cabin_type(df) #将船仓类型变为——有 无
    dummies_Cabin = pd.get_dummies(df['Cabin'], prefix= 'Cabin')
    dummies_Embarked = pd.get_dummies(df['Embarked'], prefix= 'Embarked')
    dummies_Sex = pd.get_dummies(df['Sex'], prefix= 'Sex')
    dummies_Pclass = pd.get_dummies(df['Pclass'], prefix= 'Pclass')
    dummies_Name = set_Name_type(df)

    df = pd.concat([df, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass, dummies_Name], axis=1)
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)

    # 用preprocessing模块做scaling
    scaler = preprocessing.StandardScaler()
    #df.fillna(0.0, inplace = True)  #填充所有缺失数据
    df.Age = df.Age.fillna(value=0.0)
    age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1), age_scale_param)
    df.Fare = df.Fare.fillna(value=0.0)
    fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1,1))
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1), fare_scale_param) 
    df.SibSp = df.SibSp.fillna(value=0.0)
    scale_param = scaler.fit(df['SibSp'].values.reshape(-1,1))
    df['SibSp_scaled'] = scaler.fit_transform(df['SibSp'].values.reshape(-1,1), scale_param) 
    df.Parch = df.Parch.fillna(value=0.0)
    scale_param = scaler.fit(df['Parch'].values.reshape(-1,1))
    df['Parch_scaled'] = scaler.fit_transform(df['Parch'].values.reshape(-1,1), scale_param) 

    return df

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

def set_Name_type(df):
    ss = df['Name'].astype(str)
    s1 = pd.get_dummies(ss.str.contains('Mr.',na=False), prefix= 'Name_Mr') #将含有固定字符的设置为True，并将其转换为2个字段
    s2 = pd.get_dummies(ss.str.contains('Mrs.',na=False), prefix= 'Name_Mrs') 
    s3 = pd.get_dummies(ss.str.contains('Miss.',na=False),prefix= 'Name_Miss') 
    dummies_Name = pd.concat([s1, s2, s3], axis=1)
    return dummies_Name

def load_data_wrapper():

    #读取训练数据
    train_data = load_data(0)
    #将特征值取出并转换为矩阵
 #   train_df = train_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*|Mr|Mrs|Miss')
    train_df = train_data.filter(regex='Survived|Age_scaled|SibSp_scaled|Parch_scaled|Fare_scaled|Cabin_.*|Sex_.*|Pclass_.*|Name_.*')
    #print(train_df.columns) # 打印列索引
    train_np = train_df.as_matrix()
    # print(train_df.dtypes) #查看不同列的数据类型
    # y即Survival结果
    trd_y = train_np[:, 0]
    # X即特征属性值
    trd_X = train_np[:, 1:]

    #读取竞赛数据
    test_data = load_data(1)
    #将特征值取出
    test_df = test_data.filter(regex='Age_scaled|SibSp_scaled|Parch_scaled|Fare_scaled|Cabin_.*|Sex_.*|Pclass_.*|Name_.*')
    test_np = test_df.as_matrix()
    # y即Id
    test_y = test_data['PassengerId'].as_matrix()
    # X即特征属性值
    test_X = test_np

    return trd_X, trd_y, test_X, test_y

def save_data(X, y):
    result = pd.DataFrame({'PassengerId':y.astype(np.int), 'Survived':X.astype(np.int32)})
    result.to_csv("Titanic/data/result.csv", index=False)

def save_error(X):
    with open('Titanic/data/error.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        #writer.writerow(["PassengerId","Survived"])
        #写入多行用writerows
        writer.writerows(X)


load_data_wrapper()


def save_data0(X, y):
    with open('Titanic/data/result.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["PassengerId","Survived"])
        #写入多行用writerows
        writer.writerows(X, y)
        

'''
查找出错项
origin_data_train = pd.read_csv("/Users/HanXiaoyang/Titanic_data/Train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:,0]]['PassengerId'].values)]
bad_cases

'''