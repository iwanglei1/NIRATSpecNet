# 代码实现

- test下为数据集
- .ipyb 后缀文件是PLS、SVR和ANN代码
- .png 后缀文件是可视化之后的原始结果
- Model.py 文件可用来加载CY.h5(柴油模型)、YM.h5(玉米模型)和NN(牛奶模型)等
- regulating_Hyperparameter.py 是模型结构

- 随机初始化会影响模型结果，使用我训练好的模型即可复现论文结果（后缀为.h5 的模型），尽量使用下文所示环境

- 硬件环境 操作系统 windows10、显卡 Tesla V100 32G
- 软件环境 python==3.6 、Keras==2.3 、Tensorflow==2.0