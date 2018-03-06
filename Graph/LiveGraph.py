"""在线绘图 即非阻塞连续绘图"""

from pygal import Line
import sys
from os import path
from subprocess import Popen,PIPE
import webbrowser
serverName="GraphServer.py"
end_chars="__end__"
class LiveGraph():
    def __init__(self,fname="Figure",port=1000):
        self.name=fname
        self.plt_dict={}
        #得到当前脚本目录地址
        self.dir=path.split(__file__)[0]
        #得到server地址
        self.server_file=path.join(self.dir,serverName)
        #启动服务器
        self.server=Popen(["python",self.server_file,str(port)],stdin=PIPE)
        #次数记录
        self.count=0
        #打开浏览器
        print(f"启动服务器 地址:http://localhost:{port}")
        # webbrowser.open(f"http://localhost:{port}")
    def log(self,name,y):
        """记录一个y"""
        self.count+=1
        if name in self.plt_dict:
            lst=self.plt_dict[name]
        else:
            lst=[]
            self.plt_dict[name]=lst
        lst.append(y)
        #折线图
        tl=Line()
        #画出x轴表示记录次数 最大画20格
        # tl.x_labels=range(0,self.count,max(1,self.count//20))
        #绘图
        for k in self.plt_dict:
            tl.add(k,self.plt_dict[k])
        str=tl.render(is_unicode=True)+f"\n{end_chars}\n"
        self.server.stdin.write(str.encode())
    def __del__(self):
        self.server.terminate()