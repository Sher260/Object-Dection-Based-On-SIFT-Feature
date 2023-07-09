from sklearn.svm import LinearSVC,SVC
import joblib
import matplotlib.pyplot as plt
import numpy as np
import time as tm


# 分类模型地址
model_path = 'C:/projects/svm_detection-master/svm_model_fenlei'

# 加载特征和标签
N_train_img = 0 # 训练图片
N_train_block = N_train_img * 60 # 训练的块数量
fds = np.load('data_x_playground.npy')
labels = np.load('data_y_playground.npy')

# 将数据随机打乱
N_block_all = len(labels)  # 总的块的数量
ID = np.random.choice(list(range(N_block_all)), N_block_all)# 乱序排列
fds = fds[ID,:]
labels = labels[ID,:]

# 训练数据
x_train = fds[:N_train_block, :]
y_train = labels[:N_train_block, 0]
# 测试数据

x_test = fds[N_train_block:, :]
y_test = labels[N_train_block:, 0]

# 评价指标（20次分类统计
R = np.zeros(20)
Pre = np.zeros(20)
Acc = np.zeros(20)
FA = np.zeros(20)
MA = np.zeros(20)
FA_block = np.zeros(20)
Rall = 0
Pall = 0
Aall = 0
FAall = 0
MAall = 0

t1 = tm.time()
# 定义分类器
# clf = LinearSVC(class_weight={1: 110}, max_iter=5000)  # 设置权重比 正：负
clf = SVC(class_weight={1: 20}, max_iter=500000)  # 设置权重比 正：负
for i in range(20):
    [posive_ID] = np.where(y_train==1)  # 正样本
    [negative_ID] = np.where(y_train==0)  # 正样本
    negative_ID = np.random.choice(negative_ID, 70 * posive_ID.size)  # 正负样本数量比： 1:70
    all_ID = np.append(posive_ID, negative_ID)
    x_train1 = x_train[all_ID, :]
    # x_train1 = scaler.fit_transform(x_train1)
    y_train1 = y_train[all_ID]
    model_path = 'C:/projects/svm_detection-master/svm_model_fenlei'
    # clf.fit(x_train1, y_train1)  # 网络训练
    # joblib.dump(clf, model_path)  # 保存模型

    x_input = x_test
    y_input = y_test
    clf = joblib.load(model_path)

    pred = clf.predict(x_input)  # 模型预测

    P = np.sum(y_input == 1)  # 总目标数量 P
    N = np.sum(y_input == 0)  # 总杂波数量 N
    TP = np.sum(pred[y_input == 1])  # 检测到的目标 TP
    FP = np.sum(pred[y_input == 0])  # 杂波虚警个数
    TN = N - FP
    R[i] = TP / P  # 召回率 R
    Pre[i] = TP / (TP + FP)  # 精确度
    Acc[i] = (TP + TN) / (P + N)  # 准确率
    MA[i] = 1 - R[i]  # 漏警率
    FA[i] = FP / (TP + FP)  # 虚警率
    FA_block[i] = FP / (x_input.shape[0] / 60)
    Rall = Rall + R[i]
    Pall = Pall + Pre[i]
    Aall = Aall + Acc[i]
    MAall = MAall + MA[i]
    FAall = FAall + FA[i]
a = 1
print('R=', Rall*5, '\n', 'Pre=', Pall*5, '\n', 'Acc=', Aall*5, '\n', 'MA=', MAall*5, '\n', 'FA=', FAall*5)
t2 = tm.time()
print(f'time:{(t2 - t1) / 60}')
x_axis_data = range(20)  # x
y_axis_data1 = R  # y
y_axis_data2 = Pre
y_axis_data3 = Acc
y_axis_data4 = MA
y_axis_data5 = FA

plt.plot(x_axis_data, y_axis_data1, 'b*--', alpha=0.5, linewidth=2, label='R',marker = 's',markersize = 4)
plt.plot(x_axis_data, y_axis_data2, 'rs--', alpha=0.5, linewidth=2, label='Pre',marker = 'o',markersize = 4)
plt.plot(x_axis_data, y_axis_data3, 'go--', alpha=0.5, linewidth=2, label='Acc',marker = '^',markersize = 4)
plt.plot(x_axis_data, y_axis_data4, 'k--', alpha=0.5, linewidth=2, label='MA',marker = '+',markersize = 4)
plt.plot(x_axis_data, y_axis_data5, 'tab:purple', alpha=0.5, linewidth=2, label='FA',marker = '*',markersize = 4)

plt.yticks(fontsize=20)
plt.legend(fontsize=20)  # 显示上面的label
plt.xlabel('time',fontsize=20)
plt.ylabel('result',fontsize=20)  # accuracy

plt.xticks(range(0, 20),fontsize=20)
plt.show()