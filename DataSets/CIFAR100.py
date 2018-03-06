"""CIFAR10数据集"""
from mxnet.gluon import data as mxdata

from experiment_frame.Base.Data import ImageDataSet

#数据集类数
num_class=100



class DataSet(ImageDataSet):
    def __init__(self,*args,**kwargs):
        super(DataSet, self).__init__(mxdata.vision.CIFAR100,num_class=num_class,*args,**kwargs)