"""提供DataLoader"""

from abc import abstractmethod

from mxnet import ndarray, image, cpu, gpu
from mxnet.gluon import data as mxdata

"""数据工具箱 包括数据源和"""
import os

import numpy as np
from abc import ABCMeta
#此为自带测试数据的位置
datapath=r"./data"
#以下所说的绝对路径 可以是真正的绝对路径 也可以是相对于当前工作目录的相对路径 不过都没有区别
def askPath():
    """询问路径"""
    print("input data path:")
    df = input()
    return df
def load(path=None):
    """简单的npz数据加载器 从测试用data目录加载文件，输入文件名 返回 data,tar"""
    prepath=os.getcwd()
    os.chdir(datapath)
    if path is None:
        df=askPath()
    else:
        df=path
    npz = np.load(df)
    data, tar = npz["X"], npz["Y"]
    #还原工作目录
    os.chdir(prepath)
    #确保为单精度浮点数
    data,tar=data.astype("float32"),tar.astype("float32")
    return data,tar
#数据集标准接口
class AbsDataSet():
    __metaclass__=ABCMeta
    def __init__(self,batch_size=None):
        self.batch_size=batch_size
    @abstractmethod
    def train(self,one_hot=True,flatten=True)->mxdata.DataLoader:pass
    @abstractmethod
    def test(self,one_hot=True,flatten=True)->mxdata.DataLoader:pass
    @abstractmethod
    def get_num_class(self):pass
    def get_batch_size(self):
        if self.batch_size is None:
            raise Exception("错误，Batchsize设置未知，且数据集不支持内置batchsize")
        return self.batch_size

class SplitDataSet(AbsDataSet):
    """将一个DataSet分割成两个dataloader"""
    def __init__(self,dataset:mxdata.Dataset,num_cls=None,test_ratio=0.2,batch_size=None):
        super().__init__(batch_size)
        self.test_ratio=test_ratio
        self.test_count=int(len(dataset)*test_ratio)
        testdata=dataset[0:self.test_count]
        traindata=dataset[self.test_count:]
        #构建dataset
        self.trainset=mxdata.ArrayDataset(*traindata)
        self.testset=mxdata.ArrayDataset(*testdata)
        #如果num_cls为None则自动测量
        if num_cls is None:
            self.num_cls=len(testdata[1])
        else:
            self.num_cls=num_cls

    def get_num_class(self):
        return self.num_cls

    def train(self, one_hot=True, flatten=True):
        """
        这里所有参数都不起作用
        :param one_hot:
        :param flatten:
        :return:
        """
        return mxdata.DataLoader(self.trainset,self.get_batch_size())

    def test(self, one_hot=True, flatten=True):
        return mxdata.DataLoader(self.testset,self.get_batch_size())

class LoaderSplitDataSet(SplitDataSet):
    """分割Loader"""
    def __init__(self, dataloader:mxdata.DataLoader, num_cls=None, test_ratio=0.2, batch_size=None):
        super().__init__(dataloader._dataset, num_cls, test_ratio, batch_size)

class DataSetSplit(LoaderSplitDataSet):
    """将一个原有数据集的train部分分割为两部分 一部分用作训练一部分用作实时验证 同时提供一个最终的real验证集"""
    def __init__(self, dataset:AbsDataSet,test_ratio=0.2, batch_size=None):
        #注意这里直接使用默认参数 如需自定义参数需要自己通过lambda替换成员函数
        super().__init__(dataset.train(), dataset.get_num_class(), test_ratio, batch_size)
        self.realtest=dataset.test()
        self.batch_size=dataset.get_batch_size()
    def real_test(self):
        """真实验证集"""
        return self.realtest



from mxnet import nd
class ImageDataSet(AbsDataSet):
    """Image数据集的通用Dataset"""

    def get_num_class(self):
        return self.num_class


    def __init__(self, cls, num_class,size=None,*args,**kwargs):
        """
        一个图片数据集
        :param cls: vision module中的一个类例如mxdata.vision.MNIST
        :param n_bsize: batch size
        :param num_class: 类数量
        :param size:目标缩放尺寸 [height*width]
        """
        super(ImageDataSet, self).__init__(*args,**kwargs)
        self.cls = cls
        self.num_class = num_class
        self.target_size=size

    def get_transformer(self,flatten=True, one_hot=True):
        """获取转换器函数"""

        def transform(data:nd.NDArray, label: nd.NDArray):
            """
            转换函数 得到0-1之间的图像和onehot型的label
            :param data:
            :param label:
            :return:
            """
            def getonehot(tlabel):
                if one_hot:
                    olabel = np.zeros(shape=(self.num_class,),dtype=np.float32)
                    olabel[tlabel] = 1
                else:
                    olabel = tlabel
                return olabel
            if type(label)==int or type(label)==np.int32:
                olabel=getonehot(label)
            else:
                olabel=np.array([getonehot(l) for l in label])
            olabel=nd.array(olabel)
            # dt转换
            dt = data.astype(np.float32) / 255
            #转换大小 这个转换极端的tmd低效 我希望
            if not (self.target_size is None):
                dt=image.imresize(dt.as_in_context(cpu()),w=self.target_size[1],h=self.target_size[0]).as_in_context(gpu()).astype(np.float32)
            l = len(dt.shape)
            if flatten:
                if l==3:
                    dt = dt.reshape((data.shape[0] * data.shape[1],))
                else:
                    #带batch
                    dt = dt.reshape((data.shape[0],data.shape[1] * data.shape[2]))
            else:
                if l==3:
                    dt=dt.reshape((dt.shape[2],dt.shape[0],dt.shape[1]))
                else:
                    dt=dt.reshape((dt.shape[0],dt.shape[3],dt.shape[1],dt.shape[2]))
            return dt, olabel

        return transform

    def train(self,one_hot=True, flatten=True):
        """通过各参数获取图片数据集的loader 懒加载 获取tuple[train,test]"""
        transform = self.get_transformer(flatten=flatten, one_hot=one_hot)
        # 准备数据
        train_set = self.cls(train=True, transform=transform)
        # 准备加载器
        train_loader = mxdata.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        return train_loader

    def test(self,one_hot=True, flatten=True):
        """通过各参数获取图片数据集的loader 懒加载 获取tuple[train,test]"""
        transform = self.get_transformer(flatten=flatten, one_hot=one_hot)
        # 准备数据
        train_set = self.cls(train=False, transform=transform)
        # 准备加载器
        train_loader = mxdata.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        return train_loader


#加载各种数据集

#加载数据集表
# class LazyDict(dict):
#     """懒加载字典用于数据集延迟加载"""
#     def __getitem__(self, k):
#         ds=super().__getitem__(k)
#         obj=ds()
#         return obj


data_loaders={}
import experiment_frame.DataSets as dts
for n in dir(dts): #type:str
    if not n.startswith("__"):
        data_loaders[n]=dts.__dict__[n].DataSet
        data_loaders[n].__name__=n
