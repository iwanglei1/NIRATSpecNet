import scipy.io as scio
import numpy as np

def getOdataMilk():
    dataFile = 'C://Users//Liangyi//Desktop//'
    dataName = 'milk.mat'
    dataProp = np.empty(67,dtype=float)
    dataMilkLength = 0
    data = scio.loadmat(dataFile + dataName)
    dataSpecimen = data['X']
    dataProp = data['y']
    dataSpecimen -= dataSpecimen.mean(axis=0)
    dataSpecimen /= dataSpecimen.std(axis=0)

    return dataSpecimen,dataProp








if __name__ == '__main__':
    getOdataMilk()
