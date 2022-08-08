# ipynb文件

* 基于朴素贝叶斯的情感分析.ipynb为项目的主文件，介绍了项目的具体内容，包含具体代码以及运行结果等。


* 朴素贝叶斯分类器模块详解.ipynb对使用的替换库中的函数进行了详细说明。

# py文件

* naive_bayes.py为本项目的完整程序，但因调用的第三方模块并非IPython提供，所以无法用pip工具直接下载，如想顺利运行程序，<font color='red'>请将替换库文件夹中的bayes文件夹完整复制到/用户/anaconda3/Lib/site-packages文件夹中</font>。

# 数据

* waimai.csv为从网站上下载的原始数据文件。


* bad.txt和good.txt为主程序直接调用的数据文件，这两个文件由waimai.csv直接生成。

# 替换模块

* 主程序没有使用IPython官方提供的朴素贝叶斯分类器模块，而是使用针对新手的ML学习包中的Bayes.py作为第三方模块。


* Bayes.py在\替换模块\bayes中。


* ipynb文件中的朴素贝叶斯分类器模块详解.ipynb对使用的替换模块中的函数进行了详细说明。

# 停用词表

* 主程序进行中文分词时使用的停用词表是文件中的cn_stopwords。
