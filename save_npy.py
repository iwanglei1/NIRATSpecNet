from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt
import someFunction as sF
import tensorflow as tf
from keras.models import load_model

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

model = load_model('my_model.h5')

chengfenshui,d5,dp5,dp6 = sF.getOdata()   #调用函数返回原始数据，数据已处理为均值为零，方差为1
test_data,test_lable,train_data,train_lable = sF.getTestData(chengfenshui,d5,dp5,dp6)
# result = model.predict(test_data)
# np.save("yuanshishujv.npy",result)
result_trian = model.predict(train_data)
np.save("xunlianji.npy",result_trian)
# print(result)
# print(type(result))
# print(result.shape)