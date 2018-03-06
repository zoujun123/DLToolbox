"""绘图监视服务器 被LiveGraph启动并不断读取LiveGraph提供的数据转换为svg图形提供给前端"""

from pygal import Line
from threading import Thread
from bottle import template,run,route,static_file
from os import path
#使用异步多线程服务器
# from gevent import monkey
# monkey.patch_all()

#获取模板文件地址
#目录地址
filedir=path.split(__file__)[0]
svgtext= "未写入"
end_chars="__end__"
def read():
    """
    不断从管道读取代码执行 每行执行都得到一串图片的base64 datauri
    :return: 无
    """
    global svgtext
    while True:
        store=""
        while True:
            #输入
            img=input()
            #检测img的内容 如果为end_chars就结束
            if img==end_chars:
                break
            else:
                #否则就在svgtext上添加一行
                store+=img
        svgtext=store


@route("/img")
def graph():
    """
    服务器函数 合成html并返回
    :return:
    """
    return svgtext
@route("/<filename>")
def static(filename):
    return static_file(filename,root=path.join(filedir,"./static"))
@route("/")
def root():
    return static(filename="Figure.html")
if __name__=="__main__":
    #读取端口参数
    import sys
    port=int(sys.argv[1])
    #启动读取线程
    th=Thread(target=read,daemon=True)
    th.start()
    #启动服务器
    run(quiet=True,port=port)
