"""测试工具箱"""
from ..common import *
from ..Base.Model import IModel,Model,IPredictable
from ..Trainer.BaseTrainer import TrainInfo
from mxnet.gluon import Trainer
from ..Base.Data import AbsDataSet
def get_tf_from_Model(name:str,model:Model,learning_rate=0.01,optimizer="rmsprop",wd=0,clip=0,testiters=3):
    optp={"learning_rate":learning_rate}
    if wd>0:
        optp["wd"]=wd
    if clip>0:
        optp["clip_gradient"]=clip
    tr=Trainer(model.collect_params(),"rmsprop",optimizer_params=optp)
    tf=TrainInfo(model,tr,print_period=100,modname=name,keep_stale=True,test_iter=testiters,train_count_on_batch=1)
    return tf

def get_tf(*args,**kwargs):
    return get_tf_from_Model(name="测试模型",*args,**kwargs)

#cross_validation
from typing import Callable,List
from .CrossValidation import ROCKcv,CrossValidation
from ..Trainer import FlowTrainer,SyncTrainer

def roc_cross_validation(model:IPredictable,func: Callable[[AbsDataSet], None],ds:AbsDataSet):
    """带ROC曲线绘制的交叉验证"""
    ROCKcv(ds,model,splitcount=3).cross_validation(func)

def cross_validation(func: Callable[[AbsDataSet], None],ds:AbsDataSet):
    "不带ROC曲线绘制的交叉验证"
    CrossValidation(ds,splitcount=3).cross_validation(func)

from .Evaluator import Evaluator
#针对具体模型
def cross_validation_infos(infos:List[TrainInfo],ds:AbsDataSet,epochs=5,method='flow',roc_pos=None,loadfile=False):
    """
    交叉验证一个模型组
    :param infos: 训练信息组
    :param ds: 数据集
    :param method: 训练方法
    :param roc_pos:绘制哪一info的模型的roc曲线 为None表示不绘制
    :param loadfile:是否全部从文件中加载参数 如果为False  则以info中的设置数据为准
    :return: 无
    """
    #确定是否需要从文件中加载参数
    if loadfile:
        for info in infos:
            info.load()
        #输出当前加载参数的性能
        Evaluator(graph=False).print_eval(info.model,ds.test())

    #训练函数
    def train(dt:AbsDataSet):
        #初始化
        for info in infos:
            mod=info.model #type:Model
            mod.initialize_model(force=True)
        #确定训练函数
        fls = {"flow": FlowTrainer.FlowTrainer(epochs=epochs, dataset=dt),
               "sync": SyncTrainer.SyncTrainer(epochs=epochs, dataset=dt)}
        fl = fls[method]
        #训练
        fl.train(infos)
    #条件满足就绘制ROC曲线
    assert roc_pos is None or roc_pos<len(infos)
    if roc_pos is not None:
        roc_cross_validation(infos[roc_pos].model,train,ds)
    else:
        cross_validation(train,ds)

    #保存所有模型
    for info in infos:
        info.save()

def cross_validation_model(name:str,model:Model,ds:AbsDataSet,epochs=5,loadfile=False,*args,**kwargs):
    """
    交叉验证单个模型
    :param name:模型名
    :param model:模型对象
    :param ds:数据集
    :param epochs:训练次数
    :param args:其他Traininfo
    :param kwargs:其他Traininfo
    :return:
    """
    info=get_tf_from_Model(name=name,model=model,*args,**kwargs)
    cross_validation_infos([info],ds,epochs=epochs,loadfile=loadfile)

