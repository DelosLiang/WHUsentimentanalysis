import xlrd

def open_excel(file_xls):#定义将excel的内容转移到列表的函数
    wb = xlrd.open_workbook(file_xls)# 打开Excel文件
    sheet = wb.sheet_by_name('Sheet2')#通过excel表格名称(rank)获取工作表
    data = []  #创建空list作为目标文件
    for r in range(sheet.nrows):#循环读取表格内容（每次读取一行数据）
        data1=[]
        for c in range(sheet.ncols):
            data1.append(sheet.cell_value(r,c))
        data.append(list(data1))
    return data

def find_excel(root_path):#root_path即要搜索的文件根目录
    import os
    
    all_file_list =[]#创建储存xls文件地址的列表

    flie_type=""#储存文件的拓展名的字符串变量
    file_data=[]#用于储存root_path下所有文件的列表
    
    file_data=os.listdir(root_path)#寻找目录下所有文件
    
    #对每个文件循环判定是否为xls类型
    for file in file_data:
        file_type=os.path.splitext(file)[1]#提取拓展名
        if file_type ==".xls":
            all_file_list.append(root_path+"\\"+file)
    return(all_file_list)#筛选特定目录下xls文件并且将文件路径做成列表

source_list=[]#八个文件的路径列表
source_list=find_excel(r'..\NRC词典\onefileperemotion')

def combine_list(s_list,target_xls):#分别为要合并的文件表与合并后的文件列表
    import xlsxwriter
    
    word_data=[]
    file_word=[]#八个文件包含的所有词语的列表
    data=[]#存放合并后的词语情感向量列表
    
    #创建词语情感向量列表的框架（输入情感词汇），提取八个词典中包涵的所有的词语到file_word中
    for file_xls in s_list:
        file=open_excel(file_xls)#运用了前文的excel读取函数
        for v in file:
            file_word.append(v[0])
    file_word=list(set(file_word))#去除file_word中的重复元素
    
    #将词汇数据移动到data，构造一个不含向量数据的二维词汇列表框架
    for word in file_word:
        data.append([word])

    
    #将不同情感的数据合并并且向量化
    for file_xls in s_list:
        file=open_excel(file_xls)
        #创建情感文件中对应的词汇索引目录，共下面索引
        word_in_file=[]
        for v in file:
            word_in_file.append(v[0])
        for m in range(len(data)):
            if data[m][0] in word_in_file:
                weight=file[word_in_file.index(data[m][0])][1]#寻找对应情感列表中相应词语(data[m][0])的权重
                data[m].append(weight)
            else :
                data[m].append(0)#遍历data(file_word)中词汇，将每种情感文件中的数据输入data列表(每个情感文件中不包括的词语加权为0)。
    
    #定义对应情感的标签(表头)
    english_df=[['English (en)','Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']]
    data=english_df+data#添加表头
    
    # 将data中的数据复制到target_xls的工作表中
    workbook = xlsxwriter.Workbook(target_xls)
    worksheet = workbook.add_worksheet()#打开目标存放文件中的工作表
    for i in range(len(data)):
        for j in range(len(data[i])):
            worksheet.write(i, j, data[i][j])
    workbook.close()


    target_list=r"..\NRC词典\all_word_emotion\all_emotions.xlsx"
combine_list(source_list,target_list)



