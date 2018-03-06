"""将一个已经训练好的模型作为参照 训练另一个模型"""

from .BaseTrainer import *
from .FlowTrainer import *
from .SequentialTrainer import *
from .SyncTrainer import *
class AbsDistillTrainer(BaseTrainer):
    def __init__(self,label_model:IModel,*args,**kwargs):
        super(AbsDistillTrainer, self).__init__(*args, **kwargs)
        self.label_model=label_model
    def train_on_batch(self, traininfo:TrainInfo,data,label):
        """在一个batch上训练一次模型 """
        #得到label模型预测数据
        #调试代码
        # bacc,bcro=self.label_model.evaluation(data,label)
        # print(f"label模型正确率:{bacc} 交叉熵:{bcro}")
        predict=self.label_model.predict(data)
        return super(AbsDistillTrainer,self).train_on_batch(traininfo,data,predict)

#以下为特化版本
class FlowDistillTrainer(FlowTrainer,AbsDistillTrainer):
    """特化版本"""
    def __init__(self,*args,**kwargs):
        super(FlowDistillTrainer, self).__init__(*args,**kwargs)
    def train_on_batch(self, traininfo:TrainInfo,data,label):
        return AbsDistillTrainer.train_on_batch(self,traininfo,data,label)

class SyncDistillTrainer(SyncTrainer,AbsDistillTrainer):
    """特化版本"""
    def __init__(self,*args,**kwargs):
        super(SyncDistillTrainer, self).__init__(*args,**kwargs)
    def train_on_batch(self, traininfo:TrainInfo,data,label):
        return AbsDistillTrainer.train_on_batch(self,traininfo,data,label)

class SequentialDistillTrainer(SequentialTrainer,AbsDistillTrainer):
    """特化版本"""
    def __init__(self,*args,**kwargs):
        super(SequentialDistillTrainer, self).__init__(*args,**kwargs)
    def train_on_batch(self, traininfo:TrainInfo,data,label):
        return AbsDistillTrainer.train_on_batch(self,traininfo,data,label)