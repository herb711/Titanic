import titanic_loader
import network

# 将数据集拆分成三个集合：训练集、交叉验证集、测试集
training_data, testing_data, sub_data = titanic_loader.load_data_wrapper()
# 生成神经网络对象，神经网络结构为三层，每层节点数依次为（784, 30, 10）
net = network.Network([8, 7, 2])
# 用（mini-batch）梯度下降法训练神经网络（权重与偏移），并生成测试结果。
# 训练回合数=30, 用于随机梯度下降法的最小样本数=10，学习率=3.0
net.SGD(training_data, 30, 10, 0.1, test_data=testing_data)

net.wb_save() #记录权重和基值

result = net.evaluate2(sub_data) #读取竞赛数据
titanic_loader.save_data(result) #保存结果

