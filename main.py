import os

from nltk.parse.stanford import StanfordDependencyParser
from nltk.parse.stanford import StanfordParser
from nltk.tag import StanfordNERTagger
from nltk.tokenize import StanfordSegmenter
from nltk.tree import Tree

# 改变java环境变量
# java_path = "C:\Program Files\Common Files\Oracle\Java\javapath\java.exe"
# java_path="C:\Program Files (x86)\Common Files\Oracle\Java\javapath"
# os.environ['JAVA_HOME'] = java_path

# 新闻标题和第一段正文，在后面只对正文做处理
title = '苏伊上运河恢复通航!搁浅货轮已完全恢复至正常航道'
text = "新华社快讯埃及苏伊士运河管理局29日发布公报说搁浅货轮已经完全恢复至正常航道。23日，一艘悬挂巴拿马国旗的重型货轮在苏伊士运河新航道搁浅，造成航道拥堵。25日，苏伊士运河管理局正式宣布运河暂停航行。就在不久前，埃及总统塞西曾宣布苏伊士运河河道的疏通工作成功完成。"


def tree(x):
    x = x.replace(',', '')
    x = x.replace('Tree', '')
    x = x.replace('[', '')
    x = x.replace(']', '')
    x = x.replace('\'', '')
    return x

# 分词
segmenter = StanfordSegmenter(
    java_class='edu.stanford.nlp.ie.crf.CRFClassifier',
    path_to_jar=r"lab1/stanford-segmenter-4.2.0.jar",
    path_to_slf4j=r"lab1/slf4j-api.jar",
    path_to_sihan_corpora_dict=r"lab1/data",
    path_to_model=r"lab1/data/pku.gz",
    path_to_dict=r"lab1/data/dict-chris6.ser.gz"
)
result = segmenter.segment(text)
print(result)

# 命名实体识别
chi_tagger = StanfordNERTagger(
    model_filename=r'lab1/chinese.misc.distsim.crf.ser.gz',
    path_to_jar=r'lab1/stanford-ner-4.2.0.jar')
for word, tag in chi_tagger.tag(result.split()):
    print(word, tag)

# 句法分析
chi_parser = StanfordParser(r"lab1/stanford-parser.jar",
                            r"lab1/stanford-parser-4.2.0-models.jar",
                            r"lab1/chineseFactored.ser.gz")

x = str(list(chi_parser.parse(result.split())))
t = tree(x)

# 依存关系分析
chi_parser = StanfordDependencyParser(r"lab1/stanford-parser.jar",
                                      r"lab1/stanford-parser-4.2.0-models.jar",
                                      r"lab1/chineseFactored.ser.gz")
res = list(chi_parser.parse(result.split()))
for row in res[0].triples():
    print(row)

Tree.fromstring(t).draw()  # 输出句法树
