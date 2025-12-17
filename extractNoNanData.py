import scipy.io as scio
import numpy as np

def getOdataFuel():
    dataFile = 'test/'
    dataName = 'chaiyou.mat'
    dataVisc = np.empty(784,dtype=float)
    dataViscLength = 0
    tempDataVisc = []
    data = scio.loadmat(dataFile+dataName)
    dataspecimen = data['diesel_spec']
    dataprop = data['diesel_prop']
    dataspec = dataspecimen[0,0]['data']
    dataattr = dataprop[0,0]['data']

    for i in dataattr:
        dataVisc[dataViscLength] = i[1]          #修改i[]的索引，即可得到不同的属性值
        dataViscLength = dataViscLength+1
    # 得到dataspecimen 、 dataprop

    for i2 in range(len(dataVisc)):
        if(np.isnan(dataVisc[i2])):
            # print("当前已删除 %d"%i2)
            tempDataVisc.append(i2)

    visc = np.delete(dataVisc,tempDataVisc)
    spec = np.delete(dataspec,tempDataVisc,axis=0)
    # print(spec)
    print(len(spec))
    # 输出处理完nan之后的结果
    # 样品只有0-394
    # print(784-len(tempDataVisc))
    spec -= spec.mean(axis=0)
    spec /= spec.std(axis=0)
    return spec,visc


if __name__ == '__main__':
    getOdataFuel()
