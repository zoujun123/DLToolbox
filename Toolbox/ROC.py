
import numpy as np
from mxnet.gluon.data import DataLoader
from scipy import interp
import matplotlib.pyplot as mplt

from sklearn.metrics import roc_curve, auc

# #############################################################################
# Data IO and generation

# Import some data to play with
from experiment_frame.Base.Model import IModel, IPredictable


class ROCCurve():
    def __init__(self) -> None:
        super().__init__()
        #为一个元组列表 每个item为一个(y_true,y_pred_proba)形式的元组
        self.data=[]
    def update_data(self,y_true,y_pred_proba):
        self.data.append((y_true,y_pred_proba))
    def update(self,mod:IPredictable,dataset:DataLoader,maxiters=5):
        """
        更新数据 即用dataset测试mod 记录下所有的测试结果
        :param mod: 模型
        :param dataset: 数据集（测试集)
        :return:无
        """
        y_pred=[]
        y_true=[]
        for i,(data,label) in enumerate(dataset):
            if i>=maxiters:
                break
            t_y_pred=mod.predict(data).asnumpy()
            y_pred.extend(t_y_pred)
            y_true.extend(label.argmax(axis=1).asnumpy())
        #记录为一个元组
        self.update_data(np.array(y_true),np.array(y_pred))
    def draw_curve(self,pos_label=1,plt=None):
        """
        根据data中保存的每条数据进行绘制 分别绘制每个item 再绘制平均曲线\
        :param pos_label: 哪个表示正类 默认为1
        :return:
        """
        if plt is None:
            plt=mplt.subplot(111)
            isshow=True
        else:
            isshow=False

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0
        for y_true,probas_ in self.data:
            # 绘制ROC曲线 提供正类置信度 并指定正类id
            fpr, tpr, thresholds = roc_curve(y_true, probas_[:, pos_label],pos_label=pos_label)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1
        #绘制直线
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.set_xlim(-0.05, 1.05)
        plt.set_ylim(-0.05, 1.05)
        plt.set_xlabel('False Positive Rate')
        plt.set_ylabel('True Positive Rate')
        plt.set_title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

        #默认参数 则主动显示
        if isshow:
            mplt.show()
