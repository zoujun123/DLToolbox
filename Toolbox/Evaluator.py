"""验证工具集"""
import numpy as np
from mxnet.gluon import data as mxdata

from experiment_frame.Graph.LiveGraph import LiveGraph
from experiment_frame.Base.Model import IModel

def drawfunc(func):
    """
    装饰器 用于表示一个成员函数需要创建了Graph
    :param func: 装饰函数
    :return:
    """

    def checked(self,*args, **kwargs):
        if self.acc_graph is None:
            print("本测试器未创建Graph服务")
            return
        return func(self,*args, **kwargs)

    return checked

class Evaluator():
    # graph端口号从1000开始每个Evalution递增一次
    nowport = 1000

    def __init__(self, graph=True) -> None:
        super().__init__()
        # 绘图服务器创建
        if graph:
            self.acc_graph = LiveGraph(port=Evaluator.nowport)
            self.loss_graph=LiveGraph(port=Evaluator.nowport+1)
            self.cro_graph=LiveGraph(port=Evaluator.nowport+2)
            Evaluator.nowport += 3

    def __del__(self):
        if hasattr(self,"acc_graph") and self.acc_graph is not None:
            Evaluator.nowport-=3


    def print_eval(self, mod: IModel, dataset: mxdata.DataLoader, maxiters=None, dtname="测试集"):
        """
        在指定数据集上测试某个模型
        :param mod:符合标准的模型
        :param dataset: 数据loader
        :param maxiters: 最大轮
        :param dtname: 输出名字
        :param roc: 是否需要绘制ROC曲线
        :return: (指标名字列表，指标值列表，预测结果(y_trues,y_preds))
        """
        # 每个item表示一次测试的记录 包括N个参数
        cslist = []
        # 保存整个数据集上的预测和正确的Y 其中预测结果为概率分布
        y_trues = []
        y_preds = []
        for i, dt in enumerate(dataset):
            # 限制最大轮数
            if not (maxiters is None) and i == maxiters:
                break
            data, label = dt[0], dt[1]
            loss = mod((data, label)).mean()
            ls = loss.asscalar()
            # 计算得到各个参数 把loss插到最前面
            canshu = list(mod.evaluation(data, label))
            canshu.insert(0, ls)
            cslist.append(canshu)
        # 每个item表示一种参数
        meanlist = np.array(cslist).mean(axis=0)
        namelist = list(mod.evaluation_names())
        namelist.insert(0, "Loss值")
        for name, cs in zip(namelist, meanlist):
            print(f"{dtname}平均{name}:{cs}")
        print("========================")
        # 返回参数名字与值
        return namelist, meanlist

    @drawfunc
    def draw_params(self, mod, now_loss, train_tuple=None, test_loader=None, mod_name="", **kwargs):
        """
        绘图函数 每个batch绘制一次
        :param mod: 要衡量的模型
        :param now_loss: 当前训练的loss
        :param train_tuple: 当前train数据集
        :param test_loader: 测试集
        :param mod_name: 模型名
        :param detail: 是否绘制详情
        :return: 无
        """
        print(f"当前训练loss:{now_loss.asscalar()}")
        # 验证
        # 训练集
        if train_tuple is not None:
            data = train_tuple[0]
            label = train_tuple[1]
            meanlist = mod.evaluation(data, label)
            acc, cro = meanlist[0], meanlist[1]
            print(f"单Batch正确率:{acc} 单Batch训练集交叉熵:{cro}")
            self.acc_graph.log(f"{mod_name}-Train-Accuracy", acc)
            self.loss_graph.log(f"{mod_name}-Train-Loss", now_loss.asscalar())
            self.cro_graph.log(f"{mod_name}-Train-Crossentropy", cro)
        # 测试集
        if test_loader is not None:
            # nl ml分别为指标的名字列表和值列表
            nl, ml = self.print_eval(mod, test_loader, dtname="测试集", **kwargs)
            ls,acc, cro = ml[0], ml[1], ml[2]
            # 绘图
            self.acc_graph.log(f"{mod_name}-Test-Accuracy", acc)
            self.loss_graph.log(f"{mod_name}-Test-Loss", ls)
            self.cro_graph.log(f"{mod_name}-Test-Crossentropy", cro)
