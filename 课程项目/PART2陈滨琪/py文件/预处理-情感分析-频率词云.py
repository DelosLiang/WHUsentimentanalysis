import nltk
fn = open('计算社会科学论文集.txt','r',encoding='utf-8') 
string_data = fn.read() 
fn.close()
splitresult=nltk.word_tokenize(string_data)

nltk.download("stopwords")  
from nltk.corpus import stopwords 
stop_words=stopwords.words("english")
def remove_stop_words(splitresult,stop_words):
    return" ".join([word for word in splitresult if word not in stop_words])
removeresult=remove_stop_words(splitresult,stop_words)

import re 
def clean_text(x):
    temp=re.sub(r'([^\s\w]|_)+','',x).split()
    return " ".join(word for word in temp)
cleanoutcome=str(clean_text(removeresult))

from nltk import stem 
def get_stems(word,stemmerresult):
    stemmerresult=stem.SnowballStemmer("english")
    return stemmerresult.stem(word)
stemmeresult=stem.SnowballStemmer("english")
snowballoutcome=get_stems(cleanoutcome,stemmeresult)

nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer 
lemmatizer=WordNetLemmatizer()
def get_lemma(word):
    return lemmatizer.lemmatize(word)
wordnetoutcome=get_lemma(snowballoutcome)

with open("文本预处理后的计算社会科学论文集.txt","a+",encoding="utf-8")as f:
    f.write(wordnetoutcome)
    f.close

from textblob import TextBlob
fnoutcome = open('文本预处理后的计算社会科学论文集.txt','r',encoding='utf-8') 
data = fnoutcome.read()  
fnoutcome.close() 
blob=TextBlob(data)
print(blob.sentiment)

import sys
sys.path.append('../')
import jieba
import jieba.analyse
from optparse import OptionParser
from wordcloud import WordCloud,ImageColorGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

USAGE = ["-f","文本预处理后的计算社会科学论文集.txt","-k",500,"文本预处理后的计算社会科学论文集.txt"]    
parser = OptionParser()                     
parser.add_option("-f", dest="file_name")   
parser.add_option("-k", dest="topK")
opt, args = parser.parse_args(USAGE)       
print("op : ",opt)                          
print("ar : ",args)
if len(args) < 1:
    print(USAGE)            
    sys.exit(1)             
file_name = args[0]        

if opt.topK is None:        
    topK = 10
else:                       
    topK = int(opt.topK)
content = open(file_name, 'rb').read()                  
tags = jieba.analyse.extract_tags(content, topK=topK)  
result=(",".join(tags))                                  

mask=np.array(Image.open("头像.png")) 
wordcloud = WordCloud(mask=mask,font_path = 'OLDENGL.TTF',max_words=500,mode='RGBA',background_color=None).generate(result)
image_colors = ImageColorGenerator(mask) 
wordcloud.recolor(color_func=image_colors)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
wordcloud.to_file('wordcloud.png')








