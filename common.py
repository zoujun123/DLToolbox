"""各种工具函数 通用的引入"""


def lazy_property(function):
    attribute='_lazy_'+function.__name__
    @property
    def wrapper(self):
        if not hasattr(self,attribute):
            setattr(self,attribute,function(self))
        return getattr(self,attribute)
    return wrapper


class SimpleQueue(list):
    def __init__(self,*args,**kwargs):
        super(SimpleQueue, self).__init__(*args,**kwargs)
    def get(self):
        """返回第0个元素"""
        if len(self)==0:
            return None
        z=self[0]
        self.pop(0)
        return z
    def put(self,x):
        """放入一个元素"""
        self.append(x)
class LimitQueue(SimpleQueue):
    """限长的queue"""
    def __init__(self,max_len,*args,**kwargs):
        super(LimitQueue, self).__init__(*args,**kwargs)
        self.m_max_len=max_len
    def put(self,x):
        """限长"""
        if len(self)==self.m_max_len:
            return False
        super(LimitQueue,self).put(x)

class CircleQueue(LimitQueue):
    """循环的queue"""
    def __init__(self,*args,**kwargs):
        super(CircleQueue, self).__init__(*args,**kwargs)
    def put(self,x):
        """循环 如果满了就去掉第一个"""
        if len(self)==self.m_max_len:
            self.pop(0)
        super(CircleQueue,self).put(x)