import numpy as np
from mxnet.gluon import data as mxdata

from experiment_frame.Base.Data import load, AbsDataSet
from mxnet import nd
from imblearn import under_sampling,over_sampling
import random
def init(bsize):
    data,label=load("Kaggle.npz")
    #转换到球极坐标
    # norm=np.sqrt(np.sum(data**2,axis=1,keepdims=True))
    # ag=data/norm
    # data=np.concatenate([data,norm,ag],axis=1)
    #使用sin和cos信息
    # data=np.concatenate([np.sin(data),np.cos(data)],axis=1)

    # 下采样 制造平衡样本
    cr = under_sampling.NearMiss(version=3)
    data,label=cr.fit_sample(data,label)
    #上采样 制造平衡样本
    # ocr=over_sampling.ADASYN()
    # data,label=ocr.fit_sample(data,label)
    #混肴
    idx=list(range(len(data)))
    random.shuffle(idx)
    data,label=data[idx],label[idx]
    #onehot
    olabel=np.zeros(shape=(len(label),2))
    for i,l in enumerate(label):
        olabel[i][int(l-1)]=1

    #类型转换
    data=data.astype("float32")
    olabel=olabel.astype("float32")
    #
    train_sum=int(len(data)/1.3)
    tdata,tlabel=data[:train_sum],olabel[:train_sum]
    test_data,test_label=data[train_sum:],olabel[train_sum:]
    train_set=mxdata.ArrayDataset(nd.array(tdata),nd.array(tlabel))
    test_set=mxdata.ArrayDataset(nd.array(test_data),nd.array(test_label))


    #
    #loader
    train_loader=mxdata.DataLoader(train_set,batch_size=bsize)
    test_loader=mxdata.DataLoader(test_set,batch_size=bsize)

    return train_loader,test_loader
class DataSet(AbsDataSet):
    def __init__(self,*args,**kwargs):
        super(DataSet, self).__init__(*args,**kwargs)
        self.train_loader,self.test_loader=init(bsize=self.batch_size)
    def train(self, **kwargs):
        return self.train_loader
    def test(self,**kwargs):
        return self.test_loader
    def get_num_class(self):
        return 2