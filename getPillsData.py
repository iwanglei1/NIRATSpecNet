import scipy.io as scio
import numpy as np


def getOdatePills():
    dataFile = 'C://Users//Liangyi//Desktop//'
    dataName = 'yaopian.mat'
    dataHardness = np.empty(460, dtype=float)
    dataHardnessLength = 0
    data = scio.loadmat(dataFile + dataName)
    dataspecimen = data['test_1']
    dataprop = data['test_Y']
    dataspec = dataspecimen[0,0]['data']
    dataattr = dataprop[0,0]['data']

    for i in dataattr:
        dataHardness[dataHardnessLength] = i[1]          #修改i[]的索引，即可得到不同的属性值
        dataHardnessLength = dataHardnessLength + 1

    dataspec -= dataspec.mean(axis=0)
    dataspec /= dataspec.std(axis=0)
    # print(dataspec,dataHardness)

    return dataspec,dataHardness



if __name__ == '__main__':
    getOdatePills()