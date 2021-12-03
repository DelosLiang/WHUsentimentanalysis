# ipynb文件
- "NRC_analysis第一部分.ipynb"为项目主文件的第一部分，介绍了项目的第一部分具体内容、代码、运行结果
- "NRC_analysis第二部分.ipynb"为项目主文件的第二部分，介绍了项目第二部分的具体内容、代码、运行结果

# py文件
- "基于NRC词典的文本情感分析——第一部分.py"为项目第一部分的完整程序，运行结果参照ipynb文件
- "基于NRC词典的文本情感分析——第二部分.py"为项目第二部分的完整程序，运行结果参照ipynb文件

# 数据
- "Hi, mom.txt"为从网站上获得的电影评论数据文件
- "分析结果.xlsx"是由主程序"NRC_analysis第二部分.ipynb"经过"Hi, mom.txt"生成并且再调用的文件

# NRC词典
- "onefileperemotion"内的八个文件是由NRC官网上下载的词汇情感数据文件，是词语对应每个情感的数据文件
- "all_word_emotion.xlsx"中的"all_emotions.xlsx"是项目第一部分由"onefileperemotion"内的八个文件数据写入的文件
-  "all_word_emotion.xls"是由 "all_word_emotion.xlsx"另存为.xls格式的文件，由于新版本xlsxwriter不支持.xlsx文件写入，故变换数据文件格式

# 停用词表
- 程序中nltk库调用的停用词表是"english"。文件运用详情参照停用词表文件中的README文件。
