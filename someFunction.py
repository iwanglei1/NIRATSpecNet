import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

################################################################################
### 从桌面上的mat文件中读取数据，然后将数据处理成均值为0，标准差为1的形式
def getOdata():
    dataFile = 'C://Users//Liangyi//Desktop//test//'
    dataName = 'corn.mat'
    data = scio.loadmat(dataFile+dataName)

    datastr5 = data['m5spec']
    datastr5p = data['mp5spec']
    datastr6p = data['mp6spec']
    dataAll = data['propvals']
# print(datastr5[0,0]['data'])         #成功
    data5 = datastr5[0,0]['data']
    datap5 = datastr5p[0,0]['data']
    datap6 = datastr6p[0,0]['data']
    dataContAll = dataAll[0,0]['data']
    moisture = np.empty(80,dtype=float)
    changdu = 0
    # print("######################################################")
    #
    # print(type(dataContAll))
    #
    # print("######################################################")
    for i in dataContAll:                                               #TODO 此处重写方便返回四种成分
         moisture[changdu] = i[1]
         changdu = changdu + 1
    data5 -= data5.mean(axis=0)  # 将数据压缩到0-1之间
    data5 /= data5.std(axis=0)
    datap5 -= datap5.mean(axis=0)
    datap5 /= datap5.std(axis=0)
    datap6 -= datap6.mean(axis=0)
    datap6 /= datap6.std(axis=0)
    return moisture,data5,datap5,datap6
def drawAcc(history):
########################################################################################################
# 绘制训练精度以及验证精度
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']
    epochs = range(1, len(acc) + 1)
    plt.plot(epochs,acc,'bo',label='Training acc')
    plt.plot(epochs,val_acc,'b',label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

########################################################################################################
def drawLoss(history):
    ## 绘制训练损失和验证损失
    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_valuse = history_dict['val_loss']
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Train loss')
    plt.plot(epochs, val_loss_valuse, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#########################################################################################################
## 返回测试数据
def getTestData(chengfenshui,d5,dp5,dp6):
    train_data = d5[:60]
    test_data = d5[60:]
    train_lable = chengfenshui[0:60]
    test_lable = chengfenshui[60:80]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(60, 700, 1)
    test_data = test_data.reshape(20, 700, 1)
    return test_data,test_lable,train_data,train_lable
###########################################################################################################
## 计算RMSEC、RMSEP
def calculate_RMSE(p_value,r_value):

    cal_tem = 0
    cc_len = len(p_value)
    for i in range(cc_len):
        tem = math.pow((p_value[i] - r_value[i]),2)
        cal_tem = cal_tem + tem
    cal_cc = cal_tem/cc_len
    cal_fin = math.sqrt(cal_cc)
    return cal_fin

##########################################################################################################
## 计算决定系数R^2
def calculate_R2(p_value,r_value):
    average = r_value.mean(axis=0)
    print("平均值：",average)
    cr_len = len(r_value)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i]-average,2)
        car_tem = car_tem + temp_r
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2]-average,2)
        cars_tem = cars_tem + temp_s
    r_2 = (cars_tem/car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2
#########################################################################################################
## 平滑曲线
def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points
###############################################################################################################
## 向CSV中写入数据
def write_To_Csv(write_data):
    df = pd.DataFrame(write_data,
                      columns=['model_name', 'epochs', 'batch_size', 'RMSEC', 'R_C', 'RMSEP', 'R_P'])  # 列表数据转为数据框
    df.to_csv('new-beer.csv', mode='a', index=False, header=False)
    return
#############################################################################################################
def calculate_R21(p_value,r_value):
    average = r_value.mean(axis=0)
    print("平均值：",average)
    cr_len = len(r_value)
    car_tem = 0
    cars_tem = 0
    for i in range(cr_len):
        temp_r = math.pow(r_value[i]-average,2)
        car_tem = car_tem + temp_r
    for i2 in range(cr_len):
        temp_s = math.pow(p_value[i2]-r_value[i2],2)
        cars_tem = cars_tem + temp_s
    r_2 = 1-(cars_tem/car_tem)
    # print("cr_len长度：",cr_len)
    # print("真值减平均：",car_tem)
    # print("预测值减平均：",cars_tem)
    # print("################")
    return r_2
###################################################################################################################
# 返回测试集、验证集
def getTestDataFuel(spec,visc):
    train_data = spec[:300]
    test_data = spec[300:]
    train_lable = visc[:300]
    test_lable = visc[300:]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(300, 401, 1)
    test_data = test_data.reshape(95, 401, 1)
    return test_data,test_lable,train_data,train_lable
###################################################################################################################
## 返回药片的测试集，验证集
def getTestDataPills(spec,hardness):
    train_data = spec[:400]
    test_data = spec[400:]
    train_lable = hardness[:400]
    test_lable = hardness[400:]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(400, 650, 1)
    test_data = test_data.reshape(60, 650, 1)
    return test_data, test_lable, train_data, train_lable
###################################################################################################################
## 返回牛奶的测试集，验证集
def getTestDataMilk(spec,dataprop):
    train_data = spec[7:]
    test_data = spec[:7]
    train_lable = dataprop[7:]
    test_lable = dataprop[:7]
    train_data = train_data.astype('float32')
    test_data = test_data.astype('float32')
    train_lable = train_lable.astype('float32')
    test_lable = test_lable.astype('float32')
    train_data = train_data.reshape(60, 1557, 1)
    test_data = test_data.reshape(7, 1557, 1)
    return test_data, test_lable, train_data, train_lable
###################################################################################################################