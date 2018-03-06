"""提供GAN的训练支持"""

from ..Base.Model import IModel,SequentialModel
from mxnet import nd
class GAN(SequentialModel):
    """
    提供GAN训练支持（临时）
    基于序列模型自动切换计算流程
    """

    def __init__(self,dis_model:IModel,gen_model:IModel,dis_batchs=1,gen_batchs=3,classes=2):
        super().__init__()
        self.add(gen_model)
        self.add(dis_model)
        self.gen_model=gen_model
        self.dis_model=dis_model
        self.dis_batchs=dis_batchs
        self.gen_batchs=gen_batchs
        #类型多少
        self.classes=classes
        #记录 0表示正在训练判别器 1表示正在训练生成器
        self.now_state=0
        #分别记录当前还剩多少个batch可以用 类似时间片
        self.gb_now=gen_batchs
        self.db_now=dis_batchs
    def next(self):
        """
        state管理函数 每个batch后调用
        :return:
        """
        if self.now_state==0:
            #正在训练判别器
            self.db_now-=1
            if self.db_now<=0:
                self.db_now=self.dis_batchs
                self.now_state=1
        else:
            #正在训练生成器
            self.gb_now-=1
            if self.gb_now<=0:
                self.gb_now=self.gen_batchs
                self.now_state=0
    def predict(self, x):
        """
        预测时直接使用判别器做预测 如果要使用生成器 自己提取gen_model
        :param x: data 传入的应是判别器的输入 例如图片等
        :return:none
        """
        model=self.dis_model
        return model.predict(x)

    def __call__(self, x, *args, **kwargs):
        """
        此处视情况而定 即now_state的值决定计算流程
        :param x: data label 即真实样本和真实标签
        :param args: 其他
        :param kwargs: 其他
        :return:none
        """
        data=x[0]
        label=x[1]
        batch_size=len(data)
        if self.now_state==1:
            #如果在训练生成器则忽略传入的数据
            #产生随机数
            gdata=nd.random.normal(loc=0,scale=0.2,shape=(batch_size,100))
            #产生标签 正类onehot
            glabel=nd.zeros(shape=(batch_size,self.classes))
            glabel[:,self.pos_label]=1
            #使此loss最小化
            loss=super(self).__call__([gdata,glabel])
            return loss
        else:
            #训练判别器 需要合成样本集
            #生成伪样本和伪标签
            #通过生成器和随机数生成fake样本
            gdata = nd.random.normal(loc=0, scale=0.2, shape=(batch_size, 100))
            fakedata=self.gen_model.predict(gdata)
            #假标签 全部设置为正类
            fakelabel = nd.zeros(shape=(batch_size, self.classes))
            fakelabel[:, self.pos_label] = 1
            #得到fakeloss 由于假样本的loss要上升 此处加上负号
            fakeloss=-self.dis_model([fakedata,fakelabel])
            #得到真实loss
            realloss=self.dis_model([data,label])
            #合成总loss
            #此处要注意 可以通过常数项控制判别器的侧重点，fakeloss表示对正类的判假能力，realloss表示对判正和区分几类正样本的能力
            #两者的下降都有利于识别正类pos_label
            loss=fakeloss+realloss
            return loss


