"""训练器基类 提供共用函数"""
from abc import ABCMeta, abstractmethod
from typing import List

from time import sleep
from experiment_frame.Base.Data import AbsDataSet
from experiment_frame.Toolbox.Evaluator import Evaluator
from mxnet import autograd as ag
from mxnet import gluon
from mxnet import nd

from experiment_frame.Base.Model import IModel

from os import path
from typing import Callable
class TrainInfo():
    model_path="./"
    """信息类"""
    def __init__(self, model:IModel, trainer:gluon.Trainer,
                 print_period, modname,
                 show_traininfo=False,keep_stale=False,
                 limit_down=None, test_iter=None,
                 train_count_on_batch=1,global_eval=True,
                 modpars={},loadfile=False):
        """
        训练信息类
        :param model: 要训练的模型
        :param trainer: Trainer生成器 lambda model:trainer
        :param print_period: 测试周期
        :param modname: 模型名
        :param detail: 是否显示详情
        :param keep_stale: 是否忽略不变梯度
        :param limit_down: loss提升阈值
        :param test_iter: 测试集的迭代测试次数
        :param loadfile:如果有参数文件就加载进来
        """
        arguments=locals()
        del arguments["self"]
        self.__dict__.update(arguments)
        self.params_file=path.join(TrainInfo.model_path,f"{modname}.par")
        #是否load？ 保存标志 方便程序访问 这个标志表示是否由文件加载参数实现的初始化
        self.loaded_init=False
        if loadfile:
            if self.load():
                self.loaded_init=True
    #需要注意的是这里保存和读取的是模型的参数 不是trainer的参数
    def save(self):
        model=self.model #type:IModel
        model.save_params(self.params_file)
    def load(self):
        #如果没有指定文件就什么都不做
        if not (path.exists(self.params_file)):
            print(f"没有需要加载的参数文件({self.modname})")
            return False
        model=self.model #type:IModel
        model.load_params(self.params_file)
        return True



class BaseTrainer():
    __metaclass__=ABCMeta
    def __init__(self, epochs:int, dataset:AbsDataSet,loss_trans:Callable[[nd.NDArray],None]=None):
        """
        基础训练器
        :param epochs: 训练轮数
        :param dataset: 数据集
        :param loss_trans: loss转换函数
        """
        self.epochs=epochs
        self.dt=dataset
        self.train_dl=dataset.train()
        self.test_dl=dataset.test()
        self.loss_trans=loss_trans
        #创建测评器
        self.evaluater=Evaluator()
    def train_on_batch(self, traininfo:TrainInfo,data,label):
        """在一个batch上训练一次模型 """
        #得到参数
        model=traininfo.model
        limit_down=traininfo.limit_down
        trainer=traininfo.trainer

        modname=traininfo.modname
        keep_stale=traininfo.keep_stale
        test_iter=traininfo.test_iter
        modpars=traininfo.modpars
        train_count_on_batch=traininfo.train_count_on_batch
        # 模型训练
        loss=None
        for i in range(train_count_on_batch):
            with ag.record():
                loss = model((data, label),*modpars)
                #变换
                if not (self.loss_trans is None):
                    loss=self.loss_trans(loss)
                #平均
                loss = nd.mean(loss, axis=0)
                #下限
                if not (limit_down is None):
                    while loss.asscalar() < limit_down:
                        loss = loss * 10
            loss.backward()
            trainer.step(batch_size=self.dt.get_batch_size(), ignore_stale_grad=keep_stale)
        #返回最后一次的loss值
        return loss

    def eval_model(self,traininfo:TrainInfo,loss,train_tuple):
        """输出模型度量值"""
        #参数
        model = traininfo.model
        modname = traininfo.modname
        test_iter = traininfo.test_iter
        show_traininfo=traininfo.show_traininfo
        if not show_traininfo:
            train_tuple=None
        #输出
        print(f"{modname}模型:")
        self.evaluater.draw_params(model, loss, test_loader=self.test_dl, mod_name=modname, maxiters=test_iter, train_tuple=train_tuple)
    @abstractmethod
    def train_models(self, tlist: List[TrainInfo]):
        """实际训练一组模型"""
        pass
    def train(self,tlist:List[TrainInfo],wait_time=0,*args,**kwargs):
        """
        在训练模型前输出提示信息 训练完后输出整个测试集的结果
        :param tlist: 要训练的模型的信息列表
        :param wait_time: 开始训练前的等待时间 单位：秒
        :param args: 附加参数
        :param kwargs:附加参数
        :return: 无
        """
        from time import time,struct_time
        logfile = open("./train_log.txt", "a")
        logfile.write("\n\n========\n\n")
        #输出提示信息
        print("您正在使用 上清 编写的深度学习实验框架（based on mxnet/gluon）")
        print(f"当前使用数据集 {type(self.dt).__name__}进行训练和测试")
        print(f"共{len(tlist)}个模型")
        if wait_time!=0:
            print(f"{wait_time}秒后开始训练........")
            sleep(wait_time)
        #实际训练模型
        self.train_models(tlist,*args,**kwargs)
        #全局测试模型
        #输出训练集整体情况
        already_list=set([])
        #加入文件日志
        for info in tlist:
            if info.global_eval and info.modname not in already_list:
                print(f"{info.modname}模型结果:")
                self.evaluater.print_eval(info.model,self.train_dl,dtname="训练集(全局)")
                namelist,meanlist=self.evaluater.print_eval(info.model,self.test_dl,dtname="测试集(全局)")
                slist=[f"{name}:{value}" for name,value in zip(namelist,meanlist)]
                s="\n".join(slist)
                logfile.write(f"\n\n{info.modname}模型结果(测试集):\n")
                logfile.write(s)
                #记录以防止重复验证
                already_list.add(info.modname)
        logfile.close()
        print("训练结束！")
