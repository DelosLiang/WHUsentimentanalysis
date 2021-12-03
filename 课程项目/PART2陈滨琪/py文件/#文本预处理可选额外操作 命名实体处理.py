#额外操作 命名实体处理
#命名实体识别是指识别文本中具有特定意义的实体，主要包括：人名、地名、机构名、专有名词等。
#Chunking 是把单词组合成分块 (chunk) 的过程，可以用来查找名词组和动词组，也可以用来分割句子。
import nltk
nltk.download("maxent_ne_chunker")#下载nltk中的chunker和words资料
nltk.download("words")
#导入分析文本
# 对短句子分析：
sentence=""# 导入需要分析的文本
# 对txt文件进行分析：
fn = open('（此处输入文件名，注意文件要和py文件在一个文件夹内）.txt','r',encoding='utf-8') # 导入文本文件，一定注意编码的格式是UTF-8
string_data = fn.read()  # 读出整个文件
fn.close() # 关闭文件
from nltk import pos_tag
from nltk import ne_chunk
from nltk import word_tokenize
def get_ner(text):
    i=ne_chunk(pos_tag(word_tokenize(text)),binary=True) #将单词分块识别标注
    return[a for a in i] #返回识别标注后的单词
neroutcome=get_ner("(此处输入要分析文本的变量名）")
b=[str(i)for i in neroutcome]#把列表中的单个元素转换为字符串类型
c=" ".join(b)#生成字符串长串
#把处理好的文本生成txt文件以便下一步分析
with open("（此处输入生成后txt文件的文件名）.txt","a+",encoding="utf-8")as f:
    f.write(c)
    f.close
