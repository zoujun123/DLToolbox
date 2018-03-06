"""串接训练器"""
from typing import Callable, List

from experiment_frame.Base.Model import Model
# 工具函数
from ..Trainer.BaseTrainer import *


def get_traininfo_of_list(modellist:List[Model],creater:Callable[[Model],TrainInfo]):
    """为一个列表的模型生成训练信息"""
    infos=[creater(m) for m in modellist]
    return infos
def get_traininfo_of_childs(model:Model, creater:Callable[[Model], TrainInfo]):
    """模型的子模型作为序列分别生成自己的训练信息"""
    childs=model._children
    infos=get_traininfo_of_list(childs, creater)
    return infos

#训练器
from .BaseTrainer import *
from experiment_frame.Base.Model import Model,SequentialModel
import copy
class SequentialTrainer(BaseTrainer):
    """
    此为CasCade方式训练的训练器 需要以来CasCadeForest模型
    """
    def __init__(self,*args,**kwargs):
        super(SequentialTrainer, self).__init__(*args, **kwargs)
    def train_models(self, tlist: List[TrainInfo]):
        """
        逐步构建Cascade模型逐步级联训练
        :param tlist: 模型列表
        :return: 无
        """
        pre_model=SequentialModel()
        now_model=SequentialModel()
        #t为模型
        for t in tlist:
            mname=t.modname
            model=t.model
            print(f"{mname}模型训练中......")

            #加入now_model
            now_model.add(model)
            print(f"当前训练层:{now_model.num_layers}")
            #训练此模型
            for i in range(self.epochs):
                print(f"epoch:{i+1}")
                for j, (data, label) in enumerate(self.train_dl):
                    print_period = t.print_period
                    #用前面的层处理data
                    input_tensor=pre_model.predict(data)
                    #处理完成
                    loss = self.train_on_batch(t, input_tensor, label)
                    if j % print_period == 0:
                        #合成traininfo
                        tempinfo=copy.deepcopy(t)
                        tempinfo.model=now_model
                        self.eval_model(tempinfo, loss,train_tuple=(data,label))
            pre_model.add(model)

    def train_childs(self,mod:Model,creater:Callable[[Model],TrainInfo]):
        """
        用于训练一个模型的所有子模型（单独训练)
        :param mod:父模型
        :return:无
        """
        traininfos=get_traininfo_of_childs(model=mod,creater=creater)
        self.train_models(traininfos)
