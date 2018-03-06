"""实验中所有模型的接口类"""

import os
import shutil
from abc import abstractmethod, ABCMeta
from os import path

# 得到默认的context
from mxnet import Context
from mxnet import nd
from mxnet.gluon.nn import Block
from mxnet.metric import Accuracy, CrossEntropy
from sklearn.metrics import recall_score,precision_score

from typing import Callable
import numpy as np
ctx=Context.default_ctx

class IPredictable():
    """预测器接口"""
    __metaclass__=ABCMeta
    def __init__(self):
        # 在二分类中这个表示作为正类的类id 多分类中由于是全局统计 这个参数没有意义
        self.pos_label = 1
    @abstractmethod
    def predict(self,data):
        pass
class IModel(IPredictable):
    __metaclass__=ABCMeta
    """
    指定提供预测和验证接口 验证接口返回(accuracy,crossentropy) 预测接口返回y_pred
    同时规定所有Model的forward函数必须返回loss值 预测值返回由predict函数得到
    """
    def __init__(self):
        super(IModel, self).__init__()

    def evaluation(self,x,y_true):
        """
        输入一组数据和标签返回正确率和交叉熵(y与y_true)
        :param x: data
        :param y_true: label(one-hot-like)
        :return: (accuracy,crossentropy)
        """
        #处理onehot标签得到真实标签
        nor_label=nd.argmax(y_true,axis=1,keepdims=False) #type:nd.NDArray
        #predict不一定是softmax过的值 应将其归一化 使其相加值为1
        #否则会出现NaN的情况
        raw_pred=self.predict(x) #type:nd.NDArray
        y_pred = raw_pred/raw_pred.sum(axis=1,keepdims=True) #type:nd.NDArray
        y_pred_sparse=y_pred.argmax(axis=1,keepdims=False)
        ##开始求各参数
        acc = Accuracy()
        acc.update(labels=[nor_label], preds=[y_pred])
        acc_val = acc.get()[1]
        # 交叉熵
        cro = CrossEntropy()
        cro.update(labels=[nor_label], preds=[y_pred])
        cro_val = cro.get()[1]

        #确定average方式 如果预测值中的每个item的长度大于2表示是多分类 则使用macro方式统计 否则采用binary
        average="macro" if len(raw_pred[0])>2 else "binary"
        # Recall
        recall=recall_score(nor_label.asnumpy(),y_pred_sparse.asnumpy(),average=average,pos_label=self.pos_label)
        # 精确率
        precision=precision_score(nor_label.asnumpy(),y_pred_sparse.asnumpy(),average=average,pos_label=self.pos_label)
        # 返回
        return acc_val, cro_val,recall,precision
    def evaluation_names(self):
        """
        :return: 各个参数的名字
        """
        return "正确率","交叉熵","Recall","Precision"
    @abstractmethod
    def predict(self,x):pass
    @abstractmethod
    def __call__(self, *args, **kwargs):pass
    def save_params(self,filename,*args,**kwargs):
        """
        保存模型的参数
        :param filename: 文件名
        :param args: 附加参数
        :param kwargs: 附加参数
        :return: 无
        """
        print("保存模型功能在此模型上尚未实现")

    def load_params(self,filename,*args,**kwargs):
        """
        加载模型参数
        :param filename: 文件名
        :param args: 附加参数
        :param kwargs: 附加参数
        :return: 无
        """
        print("加载模型功能在此模型上尚未实现")

    def initialize_model(self,force=False):
        """
        初始化模型
        :param force: 是否强制初始化
        :return:无
        """
        print("模型未实现初始化接口！")
        pass


class Model(Block,IModel):
    def __init__(self,*args,**kwargs):
        Block.__init__(self,*args,**kwargs)
        IModel.__init__(self)
    __metaclass__ = ABCMeta
    """
    作为Block的模型
    指定作为可训练模型的基本接口
    """
    @abstractmethod
    def forward(self, x,*args):
        """
        前向传播
        :param x: tuple[data,label]
        :param args: 附加参数
        :return: 返回loss
        """
        pass
    def save_params(self, filename,*args,**kwargs):
        Block.collect_params(self).save(filename=filename)
    def load_params(self, filename,*args,**kwargs):
        Block.collect_params(self).load(filename,ctx=ctx,*args,**kwargs)

    def initialize_model(self, force=False):
        Block.collect_params(self).initialize(force_reinit=force)


class SequentialModel(IModel):
    def __init__(self):
        super().__init__()
        self.children=[]
    def add(self,model:IModel):
        self.children.append(model)
    def __call__(self,x, *args, **kwargs):
        """最后一层输出loss"""
        if len(self.children)==0:
            return 0
        data=x[0]
        label=x[1]
        now=data
        for model in self.children[:-1]:
            now=model.predict(now)
        return self.children[-1]((now,label))

    def predict(self, x):
        now=x
        for model in self.children:
            now=model.predict(now)
        return now
    @property
    def num_layers(self):
        return len(self.children)
    def save_params(self,filename,*args,**kwargs):
        """每个子model保存到一个文件 filename为一个目录"""
        if path.exists(filename):
            shutil.rmtree(filename)
        os.mkdir(filename)
        #model参数保存目录变到filename那里来
        for i,model in self.children:
            lpath=path.join(filename,f"{str(i)}.par")
            model.save_params(lpath)
    def load_params(self,filename,*args,**kwargs):
        """从目录里逐个加载模型"""
        if path.exists(filename):
            shutil.rmtree(filename)
        os.mkdir(filename)
        #model参数保存目录变到filename那里来
        for i,model in self.children:
            lpath=path.join(filename,f"{str(i)}.par")
            model.load_params(lpath)

    def initialize_model(self, force=False):
        for model in self.children:
            model.initialize_model(force=force)


class BlockModel(IModel):
    """此类用于封装一般的Block为一个IModel"""
    def __init__(self, lossfunc, block: Block):
        super().__init__()
        self.lossfunc=lossfunc
        self.block=block


    def predict(self, x):
        return self.block(x)

    def __call__(self, x,*args, **kwargs):
        data=x[0]
        label=x[1]
        predict=self.block(data)
        loss=self.lossfunc(predict,label)
        return loss
    def save_params(self,filename,*args,**kwargs):
        self.block.collect_params().save(filename)
    def load_params(self,filename,*args,**kwargs):
        self.block.collect_params().load(filename,ctx=ctx)

    def initialize_model(self, force=False):
        self.block.collect_params().initialize(force_reinit=force)


class NDArrayModel(IModel):
    """此类用于封装一个NDArray数组为一个model 其predict恒定为传入的data loss为指定loss函数对传入data和label的返回值"""
    def __init__(self, lossfunc, data: nd.NDArray,reinit_func:Callable[[nd.NDArray],nd.NDArray]=None):
        super().__init__()
        self.lossfunc=lossfunc
        self.data=data

        self.reinit_func=reinit_func
    def predict(self,x):
        return self.data
    def __call__(self,x, *args, **kwargs):
        data=x[0]
        label=x[1]
        loss=self.lossfunc(self.data,label)
        return loss
    def save_params(self,filename,*args,**kwargs):
        nd.save(filename,self.data)
    def load_params(self,filename,*args,**kwargs):
        nd.load(filename)

    def initialize_model(self, force=False):
        if self.reinit_func is None:
            print("reinit函数未提供")
        else:
            self.data=self.reinit_func(self.data)


#以下两个是纯粹函数 不需要保存参数
class Reshape(IModel):
    def __init__(self, target_shape):
        super().__init__()
        self.target_shape=target_shape
    def __call__(self, x,*args, **kwargs):
        raise Exception("错误，Reshape层不返回loss")

    def predict(self, x):
        return nd.reshape(x,shape=self.target_shape)
    def load_params(self,filename,*args,**kwargs):
        pass
    def save_params(self,filename,*args,**kwargs):
        pass


class Softmax(IModel):
    def __init__(self, axis=-1):
        super().__init__()
        self.axis=axis
    def __call__(self, *args, **kwargs):
        raise Exception("错误，SoftMax层不返回loss")

    def predict(self, x):
        return nd.softmax(x,axis=self.axis)
    def load_params(self,filename,*args,**kwargs):
        pass
    def save_params(self,filename,*args,**kwargs):
        pass


