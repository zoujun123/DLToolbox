"""此为对Gluon的loss函数的补充 保存了目前公用的常用的loss函数"""
from mxnet import nd

def ce_loss(y:nd.NDArray,y_true:nd.NDArray)->nd.NDArray:
    """

    :param y: n*num_cls的预测分布矩阵
    :param y_true: n*num_cls的实际分布矩阵
    :return:交叉熵 n*1对应每个样本的交叉熵
    """
    ret=y_true*(-y.log())
    #求交叉熵
    ret=nd.sum(ret,axis=1)
    return ret