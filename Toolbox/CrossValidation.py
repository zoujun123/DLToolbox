"""数据源相关"""
from typing import Callable

from matplotlib import  pyplot as plt
from experiment_frame.Base.Model import IModel,IPredictable
from ..Base.Data import DataSetSplit,AbsDataSet
from mxnet.gluon import data as mxdata
from mxnet import nd
class ValidationDataSet(DataSetSplit):
    def __init__(self, dataset: AbsDataSet, splitcount=5, batch_size=None):
        super().__init__(dataset, 1/splitcount, batch_size)
        #交叉验证次数
        self.dataset=dataset
        self.splitcount=splitcount
        self.nowcount=0
        #
        self.wholetrain=dataset.train()._dataset[:]
        self.wholedata,self.wholelabel=self.wholetrain[0],self.wholetrain[1]
        self.splen = int(len(self.wholetrain[0]) / self.splitcount)  # 一段的长度
    def next(self):
        """切换到下一状态"""
        self.nowcount += 1
        if self.nowcount>=self.splitcount:
            print("交叉验证次数已用完")
            return
        startpoint=self.nowcount*self.splen #测试集起始位置
        #这里直接引用
        testar=(self.wholedata[startpoint:startpoint+self.splen],self.wholelabel[startpoint:startpoint+self.splen]) #截取一段作为测试集
        self.testset=mxdata.ArrayDataset(*testar)
        #截取一段做训练集 将第一段复制过来并截取后续 这里由于要操作原始数据 于是创建副本
        trainar=[self.wholedata.copy(),self.wholelabel.copy()]
        ##框架BUG！！！！切片赋值只能先转到numpy数组再赋值否则无效
        trainar[0][startpoint:startpoint+self.splen]=trainar[0][:self.splen].asnumpy()
        trainar[1][startpoint:startpoint + self.splen] = trainar[1][:self.splen].asnumpy()
        trainar[0]=trainar[0][self.splen:]
        trainar[1]=trainar[1][self.splen:]
        self.trainset=mxdata.ArrayDataset(*trainar)

    def __next__(self):
        train=self.train()
        test=self.test()
        self.next()
        return train,test



class CrossValidation():
    """交叉验证器"""
    def __init__(self, dataset: AbsDataSet, splitcount=5, batch_size=None):
        self.vdataset=ValidationDataSet(dataset,splitcount,batch_size)
        print(f"使用数据集{type(dataset).__name__}进行交叉验证\n")

    def validation(self,func,vdataset:AbsDataSet):
        """实际执行验证操作的函数"""
        func(vdataset)
    def cross_validation(self,func:Callable[[AbsDataSet],None],val_start=0,val_end=2,realtest=False):
        from time import sleep
        print(f"{self.vdataset.splitcount}折交叉验证开始")
        for i in range(self.vdataset.splitcount):
            print(f"第{i+1}轮进行中")
            self.validation(func,self.vdataset)
            self.vdataset.next()
        if realtest:
            print(f"开始进行最终验证")
            func(self.vdataset.dataset)
        print("交叉验证结束！")

from .ROC import ROCCurve
class ROCKcv(CrossValidation):

    """此类进行交叉验证 并在最终绘制出复合ROC曲线 这里要提供一个唯一的模型 在完成一次验证后 测试此模型并记录数据"""
    def __init__(self, dataset: AbsDataSet,model:IPredictable,splitcount=5, batch_size=None):
        super().__init__(dataset, splitcount, batch_size)
        self.model=model
        self.drawer=ROCCurve()

    def validation(self, func, vdataset:AbsDataSet):
        super().validation(func, vdataset)
        #这里加入记录ROC曲线的数据的步骤
        self.drawer.update(self.model,vdataset.test(),maxiters=10)

    def cross_validation(self, func: Callable[[AbsDataSet], None], val_start=0,val_end=2,**kwargs):
        super().cross_validation(func)
        import math
        #这里加入绘制ROC曲线的步骤
        print("绘制ROC曲线中......")
        #最接近的行列 优先扩展行
        size=val_end-val_start
        ncol=int(math.sqrt(size))
        nrow=math.ceil(size/ncol)
        #绘制
        for i in  range(size):
            p=plt.subplot(nrow,ncol,i+1)
            self.drawer.draw_curve(pos_label=val_start+i,plt=p)
        plt.show()


