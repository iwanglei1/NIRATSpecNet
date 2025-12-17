import scipy.io as scio
import numpy as np


def getOdatebmilkDep():
    dataFile = 'test/'
    dataName = 'beer.mat'


    data = scio.loadmat(dataFile + dataName)
    dataspecimen_train = data['XXtrain']
    dataspecimen_test = data['XXtest']
    dataprop_train = data['Ytrain']
    dataprop_test = data['Ytest']


    a = len(dataspecimen_train)
    b = len(dataspecimen_test)
    print("a_len() is:",a)
    print("b_len() is:",b)
    dataspecimen_train -= dataspecimen_train.mean(axis=0)
    dataspecimen_test -= dataspecimen_test.mean(axis=0)
    dataspecimen_train /= dataspecimen_train.std(axis=0)
    dataspecimen_test /= dataspecimen_test.std(axis=0)
    dataspecimen_train = dataspecimen_train.astype('float32')
    dataspecimen_test = dataspecimen_test.astype('float32')
    dataprop_train = dataprop_train.astype('float32')
    dataprop_test = dataprop_test.astype('float32')
    dataspecimen_train = dataspecimen_train.reshape(a,576)
    dataspecimen_test = dataspecimen_test.reshape(b,576)

    return dataspecimen_test,dataprop_test,dataspecimen_train,dataprop_train














if __name__ == '__main__':
    getOdatebmilkDep()






