"""事件管理器类"""

from threading import Thread
class Event():
    """支持多线程执行的事件分发器"""
    def __init__(self,multithread=False,name=None,doc=None) -> None:
        super().__init__()
        self.name=name
        self.invokers=[]
        self.doc=doc
        self.multithread=multithread
    def __iadd__(self, other):
        assert callable(other)
        self.invokers.append(other)
    def __call__(self, *args, **kwargs):
        for fun in self.invokers:
            if self.multithread:
                Thread(target=fun,daemon=True,args=args,kwargs=kwargs).start()
            else:
                fun(*args,**kwargs)
    def __str__(self):
        return f"name:{self.name}"
    def __repr__(self):
        return f"""name:{self.name}
        doc:{self.doc}"""