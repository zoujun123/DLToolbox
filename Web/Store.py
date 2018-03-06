"""存储代理类 用于推送各种消息"""
from .Event import Event


class Store(dict):
    """
        一个类字典，对所有添加删除的操作自动监控并产生事件
        提供changed函数支持主动产生修改事件
        允许将自定义类对象的成员纳入监控范围
    """
    def __init__(self,depth:bool=False):
        """
        初始化一个Store
        :param depth:是否进行深度监控，注意深度监控可能导致性能损耗 没有开启depth时仅有内置类型的修改受到监控 depth当前不可用
        """
        self.depth=depth
        #添加事件监听器
        self.OnChanged=Event(name="修改事件",doc="OnChange(path:str)")
        self.Added=Event(name="添加事件",doc="Added(path:str)")
        self.OnDelete=Event(name="将要删除",doc="OnDelete(path:str,obj:object)")
        self.Deleted=Event(name="删除事件",doc="Deleted(path:str)")
    def changed(self,path="/"):
        """
        发起一个修改事件
        :param path: 修改路径 如果为None则表示全局更新
        :return: 无
        """
        self.OnChanged(path)
