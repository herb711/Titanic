
#### Libraries
import titanic_loader
import network
import LogisticRegression
from sklearn.model_selection import train_test_split

# 将数据集拆分成两个集合：训练集、测试集
trd_X, trd_y, test_X, test_y = titanic_loader.load_data_wrapper()
#对训练集进行拆分
trd_X0, trd_X1, trd_y0, trd_y1 = train_test_split(trd_X, trd_y, test_size=0.20, random_state=11) #将训练数据分成训练和验证

#生成逻辑回归对象 并生成模型
reg = LogisticRegression.Regression(trd_X0, trd_y0)
print('logistic',reg.evaluate2(trd_X1, trd_y1))
reg = LogisticRegression.Regression(trd_X, trd_y)
print(reg.evaluate(trd_X, trd_y))
result1 = reg.predict(test_X)
#titanic_loader.save_data(result1, test_y) #保存结果


# 生成神经网络对象，神经网络结构为三层，每层节点数依次为（784, 30, 10）
net = network.Network([trd_X0.shape[1], trd_X0.shape[1]-1, 2]) #自动输入列数 作为 输入节点数
#组合所需要的数据
training_data = net.data_zip(trd_X0,trd_y0)
testing_data = net.data_zip(trd_X1,trd_y1)
# 用（mini-batch）梯度下降法训练神经网络（权重与偏移），并生成测试结果。
# 训练回合数=30, 用于随机梯度下降法的最小样本数=10，学习率=3.0
net.SGD(training_data, 30, 10, 0.45, test_data=testing_data)
error = net.evaluate_print(training_data) #打印评估结果
titanic_loader.save_error(error) #记录不合格的数据
result2 = net.predict(test_X) #进行预测

#result_last = (result2 + result3 + result4 + result5 + result6)/5
result_last = result2
titanic_loader.save_data(result_last, test_y) #保存结果