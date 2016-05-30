import numpy as np
from neural_network import NeuralNetwork
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt


# 读取文件
fr = open("./data.csv")
data_mat = []; label_mat = []
dict_time = {}; fr.readline()
# 读取当前行
for line in fr.readlines()[4:12313]:
    cur_line = line.strip().split(',')
    time = cur_line[-1]; type = cur_line[3]
    if time in dict_time:
        dict_time[time][type] = cur_line[-2]
    else:
        dict_time[time] = {type: cur_line[-2]}
# 生成新数据集
x_arr = []; y_arr = []
for key in sorted(dict_time.keys()):
    type_value = dict_time[key]
    if len(type_value) == 4:
        type_value_arr= [float(type_value['"temp"']), float(type_value['"co2"']), float(type_value['"hum"']), float(type_value['"lx"'])]
        x_arr.append(type_value_arr)
        y_arr.append(float(type_value['"temp"']))

nn = NeuralNetwork([4,20,1], 'logistic')
x_arr = np.array(x_arr); y_arr = np.array(y_arr)
print x_arr, len(y_arr)
# data normalizate
x_arr = np.where(x_arr>0, x_arr, np.NAN)
mean_x = np.nanmean(x_arr, 0)
x_arr = np.where(x_arr>0, x_arr, mean_x)
x_min_max = np.max(x_arr, 0)-np.min(x_arr,0)
x_arr = (x_arr-np.min(x_arr,0))/x_min_max
y_arr = x_arr[:,0]
print x_arr,y_arr

x_train, x_test, y_train, y_test = train_test_split(x_arr[:-1], y_arr[1:])
print "start fitting"
nn.fit(x_train, y_train)
errors = [];times = []
for i in xrange(x_test.shape[0]):
    o = nn.predict(x_test[i])
    errors.append((x_test[i][0]-o)*x_min_max[0])
    times.append(i)
plt.plot(times, errors)
plt.show()
