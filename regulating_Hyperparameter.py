from keras import layers
from keras import models
from  keras import metrics
from keras.optimizers import RMSprop
import keras
import someFunction as sF
import tensorflow as tf
from keras.models import load_model
import datetime
import pandas as pd
def au_Exp():
    now = datetime.datetime.now()
    now_s = now.strftime("%Y-%m-%d-%H-%M-%S")
    config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
    sess = tf.compat.v1.Session(config=config)
    ## 准备几个参数，用于后续的自动化
    epochs_au = 50
    batch_size_au = 1
    jihuo = 'tanh'


    callback_list_test =[
        keras.callbacks.ModelCheckpoint(
        filepath= now_s+'.h5',      ##文件路径 存在当前路径下吧 还好找
        monitor= 'val_loss',         ## 监控指标
        save_best_only= True        ## 保持最佳模型
        )
    ]
    moisture,d5,dp5,dp6 = sF.getOdata()
    test_data,test_lable,train_data,train_lable = sF.getTestData(moisture,d5,dp5,dp6 )

    model = models.Sequential()
    model.add(layers.Conv1D(64,7,activation=jihuo,input_shape=(700,1)))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(16,7,activation=jihuo))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(16,7,activation=jihuo))
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(8,7,activation=jihuo))
    # model.add(layers.GlobalMaxPooling1D())  ## 实际效果极差1
    model.add(layers.Flatten())

    model.add(layers.Dense(16))
    model.add(layers.Dense(16))
    # model.add(layers.Dense(4))
    # model.add(layers.Dense(2))
    model.add(layers.Dense(1))

    model.summary()
    model.compile(optimizer=RMSprop(),loss='mse')
    history = model.fit(train_data,train_lable,
                            epochs=epochs_au,
                            batch_size=batch_size_au,
                            validation_data=(test_data,test_lable),
                            callbacks= callback_list_test
                            )

    sF.drawLoss(history)  ## 绘制当前的验证曲线

    model = load_model(now_s+'.h5')
    result_trian = model.predict(train_data)
    result_predict = model.predict(test_data)
    rmsec = sF.calculate_RMSE(result_trian,train_lable) ## 训练集上的RMSE
    rmsep = sF.calculate_RMSE(result_predict,test_lable)  ## 测试集上的RMSE
    r_2_t = sF.calculate_R21(result_trian,train_lable)## 训练集上的R_2
    r_2_p = sF.calculate_R21(result_predict,test_lable)## 测试集上得R_2
    # print("Root Mean Square Error of Calibrationh is : %g"%(rmsec))
    # print("训练集上得决定系数：%f"%(r_2_t))
    # print("Root Mean Square Error of Prediction is : %g"%(rmsep))
    # print("测试集上得决定系数：%f"%(r_2_p))
    ###### 下面的代码用于自动记录实验数据

    write_data=[(now_s,epochs_au,batch_size_au,rmsec,r_2_t,rmsep,r_2_p)]#需要新写入的数据
    sF.write_To_Csv(write_data)
    return rmsep,r_2_p
