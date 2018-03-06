"""同步训练器 将输入的一组模型同步训练 即在一个batch中按顺序训练多个模型"""
from .BaseTrainer import *
class SyncTrainer(BaseTrainer):
    """同步训练器"""
    def __init__(self, *args,**kwargs):
        super(SyncTrainer, self).__init__(*args,**kwargs)
    def train_models(self, tlist: List[TrainInfo]):
        """训练一组模型"""
        for i in range(self.epochs):
            print(f"epoch:{i+1}")
            for j, (data, label) in enumerate(self.train_dl):
                # 训练每个模型
                for t in tlist:
                    print_period = t.print_period
                    loss=self.train_on_batch(t,data,label)
                    if j%print_period==0:
                        self.eval_model(t,loss,train_tuple=(data,label))
