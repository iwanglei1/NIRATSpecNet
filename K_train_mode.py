import numpy as np
import someFunction as sF
import return_mode as rM
import matplotlib.pyplot as plt
chengfenshui,d5,dp5,dp6 = sF.getOdata()   #调用函数返回原始数据，数据已处理为均值为零，方差为1
chengfenshui = chengfenshui.astype('float32')
d5 = d5.astype('float32')
d5 = d5.reshape(80,700,1)
k = 4
num_val_samples = len(d5) // k
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #',i)
    val_data = d5[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = chengfenshui[i * num_val_samples: (i+1) * num_val_samples]
    partial_train_data = np.concatenate(          ## 准备训练数据，其他所有分区的数据
        [d5[:i* num_val_samples],
        d5[(i+1)* num_val_samples:]],
        axis = 0
    )
    partial_train_targets = np.concatenate(
        [chengfenshui[:i * num_val_samples],
         chengfenshui[(i+1) * num_val_samples:]],
        axis=0
    )
    model = rM.build_mode()      ##构建keras模型（已编译）
    history =  model.fit(partial_train_data,partial_train_targets,validation_data=(val_data,val_targets),epochs=num_epochs,batch_size=1,verbose=0)  ##训练模型（静默模式，verbose=0）
    #print(history.history.keys())
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)

average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
]
# print(len(average_mae_history))
plt.plot(range(1,len(average_mae_history) + 1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
smooth_mae_history = sF.smooth_curve(average_mae_history[10:])

plt.plot(range(1,len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()