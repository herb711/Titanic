

#### Libraries
# Standard library
import csv  
import pickle
import gzip

# Third-party libraries
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd  



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

[[ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 0.]
 [ 0.]
 [ 0.]
 [ 0.]]
'''


def load_data(s):
    ''' csv数据读取 '''
    if s == 0:
        df = pd.read_csv('Titanic/data/train.csv') #读取训练数据
        ''' 读取训练标签 '''
        y = np.zeros((len(df.Survived)), dtype=np.float) #创建数组
        y = df.Survived  #标签
    else:
        df = pd.read_csv('Titanic/data/test.csv') #读取竞赛数据
        ''' 读取待识别人ID '''
        y = np.zeros((len(df.PassengerId)), dtype=np.int) #创建数组
        y = df.PassengerId  #ID


    ''' 读取训练数据 '''
    X = np.zeros((len(df.PassengerId),8), dtype=np.float) #创建数组

    ''' 训练数据预处理  '''
    pclass = df.Pclass.fillna(value=3.0) #fillna()会填充nan数据
    X[:,0] = normalization_standard(pclass) #归1化 让数据分布在0～1

    #age = df.Age.fillna(value=df.Age.mean())
    age = df.Age.fillna(value=0)
    X[:,1] = normalization_standard(age)
  
    sex = (df.Sex=='male').astype('float') #将字符数值化
    X[:,2] = sex

    X[:,3] = normalization_standard(df.SibSp)
    X[:,4] = normalization_standard(df.Parch)
    X[:,5] = normalization_standard(df.Fare)

    if s == 0:
        set_cabin(df.Cabin)
    X[:,6] = normalization_standard([change_cabin(i) for i in df.Cabin])
    X[:,7] = normalization_standard([change_embarked(i) for i in df.Embarked])

    return X, y

def normalization_gaussian(d):
    ''' 进行归一化 方法为 正太分布 '''
    x = (d - np.average(d)) / np.std(d)  # 求均值 np.average()  求标准差 np.std() 
    return x

def normalization_standard(d):
    ''' 进行归一化 方法为 标准化 '''
    x = (d - np.min(d)) / (np.max(d) - np.min(d));    # 求均值 np.average()  求标准差 np.std() 
    return x

def change_cabin(e):
    ''' 将船舱的位置对应类型赋值 '''
    fp = open("Titanic/data/cabin.pkl", "rb")
    counters = pickle.load(fp)
    fp.close() 
    x = 0.0
    if e in counters:
        x = counters[e]
    else:
        pass
    return x

def set_cabin(e):
    ''' 建立一个cabin船仓的对应表 '''
    counters = {}
    x = 0.0
    for item in e:
        if item in counters:
            pass
        else:
            x += 0.001
            counters[item] = x          
    with open("Titanic/data/cabin.pkl", "wb") as fp:
        pickle.dump(counters, fp, pickle.HIGHEST_PROTOCOL)
        fp.close()

def change_embarked(e):
    x = 0.0
    if e == "S":
        x = 0.1
    elif e == "C":
        x = 0.2
    elif e == "Q":
        x = 0.3
    return x


def load_data_wrapper():

    i, o = load_data(0)

    tr_i, te_i, tr_o, te_o = train_test_split(i, o, test_size=0.25, random_state=33) #将测试数据分成几测试和验证

    training_inputs = [np.reshape(x, (len(tr_i[0]), 1)) for x in tr_i] 
    training_results = [vectorized_result(y) for y in tr_o] #转换为矩阵，为了训练的时候做矩阵的乘法
    training_data = list(zip(training_inputs, training_results))

    test_inputs = [np.reshape(x, (len(te_i[0]), 1)) for x in te_i]
    test_results = te_o #不需要转换为矩阵，只是做最后结果的比对
    testing_data = list(zip(test_inputs, test_results))

    ''' 读取竞赛数据 '''
    sub_i, sub_o = load_data(1)

    sub_inputs = [np.reshape(x, (len(sub_i[0]), 1)) for x in sub_i] 
    sub_results = sub_o # 无效矩阵
    sub_data = list(zip(sub_inputs, sub_results))

    return training_data, testing_data, sub_data

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...1) into a corresponding desired output from the neural
    network."""
    e = np.zeros((2, 1))
    e[j] = 1.0
    return e

def save_data(x):
    #np.savetxt('Titanic/data/result.csv', x, delimiter = ',') #第一个参数为文件名，第二个参数为需要存的数组（一维或者二维）。
    #numpy.loadtxt(fname)：将数据读出为array类型。
    with open('Titanic/data/result.csv',"w") as csvfile: 
        writer = csv.writer(csvfile)
        #先写入columns_name
        writer.writerow(["PassengerId","Survived"])
        #写入多行用writerows
        writer.writerows(x)


load_data_wrapper()



def load_data0():
    ''' csv数据读取 '''

    df = pd.read_csv('Titanic/data/train.csv')
    #print (df.head())

    ''' 读取训练标签 '''
    y = df.Survived  #标签
    y = y.astype(float) #转为浮点型

    ''' 读取训练数据 '''
    subdf = df[['Pclass','Sex','Age']]  #训练数据，选取其中某几列 
    
    ''' 训练数据预处理  '''
    pclass = subdf['Pclass'].fillna(value=3.0) #fillna()会填充nan数据
    pclass = pclass/3.0 #归1化

    age = subdf['Age'].fillna(value=subdf.Age.mean())
    age = age / 100.0 
  
    sex = (subdf['Sex']=='male').astype('float') #将字符数值化

    X = pd.concat([pclass, age, sex],axis=1) # axis=1增加列 axis=0增加在后面

    return X, y

def load_data_wrapper0():
    X, y = load_data0()

    training_inputs, testing_inputs, training_results, testing_results = train_test_split(X.values, y.values, test_size=0.25, random_state=33) #将测试数据分成几测试和验证
    training_data = list(zip(training_inputs, training_results))
    testing_data = list(zip(testing_inputs, testing_results))

    return training_data, testing_data

