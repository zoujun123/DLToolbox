"""顺序训练器 即一个接一个的训练trainer"""

from .BaseTrainer import *
from gluon_sf.MyModel.Forest.Cascade.CascadeForest import CascadeForest

class FlowTrainer(BaseTrainer):
    def __init__(self,*args,**kwargs):
        super(FlowTrainer, self).__init__(*args,**kwargs)
    def train_models(self, tlist: List[TrainInfo]):
        """
        顺序训练
        :param tlist:
        :param modpars:
        :return:
        """
        for t in tlist:
            mname=t.modname
            print(f"{mname}模型训练中......")
            #训练此模型
            for i in range(self.epochs):
                print(f"epoch:{i+1}")
                for j, (data, label) in enumerate(self.train_dl):
                    print_period = t.print_period
                    loss = self.train_on_batch(t, data, label)
                    if j % print_period == 0:
                        self.eval_model(t, loss,train_tuple=(data,label))
