def open_excel(file_xls):#定义将excel的内容转移到列表的函数
    import xlsxwriter
    import xlrd
    wb = xlrd.open_workbook(file_xls)# 打开Excel文件
    sheet = wb.sheet_by_name('Sheet1')#通过excel表格名称(rank)获取工作表
    data = []  #创建空list作为目标文件
    
    #循环读取表格内容（每次读取一行数据）
    for r in range(sheet.nrows):
        data1=[]
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r,c))
        data.append(list(data1))
    return data#返回列表值


#定义文本情感分析函数，返回对应文本情感权重
def NRC_analy(all_emotion_ex,text_txt):
    import numpy as np
    import xlsxwriter
    import xlrd
    from nltk import word_tokenize
    from nltk.tokenize import word_tokenize
    
    #利用nltk对文章进行分词
    filename=text_txt
    with open(filename,encoding='utf-8') as f:#打开文本文件
        mytext = f.read()#读取文本文件
    f.close()
    wordlist=list(word_tokenize(mytext))#获得分词列表
    
    #创建词语的情感向量索引目录
    all_emotion=open_excel(all_emotion_ex)#打开总词向量文件
    all_word=[]
    for v in all_emotion:#遍历NRC字典，创建索引目录
        all_word.append(v[0])
        
    #创建文章对应的情感向量
    emotion_count=[['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']]#向量标题
    emotion_count.append([0,0,0,0,0,0,0,0])#构建情感词权重统计向量变量列表并初始化
    
    #对词语列表进行遍历，加和所有词语的情感向量，获得文章的情感向量
    for word in wordlist:
        if word in all_word:
            weight=all_emotion[all_word.index(word)][1:]#通过前面建立的索引目录找到word对应的情感向量
            v1=np.array(weight)
            v2=np.array(emotion_count[1])
            emotion_count[1]=list(v1+v2)#进行向量加法操作，将文章中每个词语对应八个情感权重叠加，获得总的情感权重
        else:
            continue
            
    #对文章的情感向量进行百分比处理
    summary=sum(emotion_count[1])
    for i in range(8):
        emotion_count[1][i]=emotion_count[1][i]/summary
    return emotion_count


all_emo_list=r'C:\Users\Administrator\Desktop\基于NRC词典的文本情感分析\NRC词典\all_word_emotion\all_emotions.xls'#词语的情感向量表
target_txt=r'C:\Users\Administrator\Desktop\基于NRC词典的文本情感分析\数据\Hi, mom.txt'#目标分析文本
emotion_list=NRC_analy(all_emo_list,target_txt)



import xlsxwriter
import xlrd
import xlwt
#打开excel文件
workbook = xlwt.Workbook(encoding = 'utf-8')#文字编码格式为UTF-8
worksheet = workbook.add_sheet('My Worksheet')

#在excel中添加表头“emotion”与“weight”
worksheet.write(0,0,  'emotion')
worksheet.write(0,1,   'weight')

#双重循环写入数据到excel中
for i in range(2):
    for j in range(8):
        worksheet.write(j+1,i,  emotion_list[i][j])

workbook.save(r'C:\Users\Administrator\Desktop\基于NRC词典的文本情感分析\数据\分析结果.xlsx')



%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd

#读取存放在excel里面的文章的情感向量数据
df = pd.read_excel(r'C:\Users\Administrator\Desktop\python大作业\all_emotion.xls')

#为导出的饼图设置颜色，标签
colors=[ 'red', 'palegreen', 'peru', 'black','gold','darkgrey','orange','deepskyblue']
labels=['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

#导出饼图
df.plot(kind='pie',y='weight',legend='False',colors=colors,labels=labels)
plt.title('NRC-emotion analysis')#为饼图设置标题
plt.legend(labels, loc="best") 
plt.legend(labels, bbox_to_anchor=(1,0), loc="lower right", 
                          bbox_transform=plt.gcf().transFigure)#设置图例位置，防止覆盖饼图
plt.show()#展示饼图
df#展示情感百分比
