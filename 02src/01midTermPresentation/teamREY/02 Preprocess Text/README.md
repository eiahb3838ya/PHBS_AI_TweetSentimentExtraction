# Quick start to document preprocess

把工作路径与py和数据文件设置为相同的路径

注意：需要安装```word2number,unidecode```

## 安装```spacy```

```Python
# 在terminal里面输入
$ pip install -U spacy

# 下载小型small语言模型
$ python -m spacy download en_core_web_sm
```



## 清洗数据的逻辑

step 1: 定义preCleaning list，这个list里的元素都会被以```“ ”```替代，这些字符被认为对解释所有句子都没有实际含义，因此在这一步就被清除

step 2: 对句子的清洗（句子级别的清洗），这一步主要是将句子变得更规范，更方便机器阅读，包含去掉奇怪的空格，slang展开，缩写展开，数字都换成阿拉伯数字等

step 3: 对解析的句子基于单词的性质进行的单词级别的清洗，单词的性质包括：专有名词（比如London），名词，代词，标点，形容词etc.



## 实例化text preprocess

text指一个string, 如“This is a cat.”

在这一步，主要需要指定用于缩略词还原和slang短语还原的字典

**缩略词还原**

```Python
from ContractionMap import ContractionMap

# 初始化缩略词典
cp = ContractionMap()

# 创造一个用`分割缩略词（而不是'分割）的镜像词典
cp.dual_contractionMap()

# 在词典里增加一写元素（#FIXME：case sensitive）
cp.add_newContractionItems({"im":"i am", "Im":"I am"})

# 返回词典 rtype: dict
CONTRACTION_MAP = cp.get_contractionMap()
```

**slang短语还原**

```Python
import pandas as pd
# 从txt文件读取slang短语
slangText = pd.read_table("slangDict.txt", delimiter="=", header=None)

# 生成slang短语的字典
SLANG_DICT = {slangText.iloc[i][0]:slangText.iloc[i][1] for i in slangText.index}

# 增加一些新的元素（注意，键值key必须全部大写）
SLANG_DICT.update({"R":"Are"})
```

下一步是**实例化TextPreprocess对象**

```Python
from TextPreprocess import TextPreprocess

# 实例化， 默认参数：语言模型 model = "en_core_web_sm"
tp = TextPreprocess()

# 选择清洗完的数据结构
tp.set_returnTokenStatus(False)

# 设置参数， deselectWordList(从stopWords字典中删除的元素)，
# contractionMap(指定缩略词字典)
# slangDict(指定slang短语字典)
# addNewStopWords = [](添加新的stopwords，默认为空)
tp.auto_setUp(preCleaningList = ["#","_"],
              deselectWordList = ["no","not"],
              contractionMap = CONTRACTION_MAP,
              slangDict = SLANG_DICT,
              notDeletePunctuation = ["!","?","..."],
              maximumPunctRepeatition = 1)
```

**例子**

```Python
# ex 1
tp.set_text("Sooo SAD I'll miss you here in San Diego!!! r u okay")
print(tp.tokenize_text())

# ex 2
tp.set_text("Journey!? Wow... u just became cooler.  hehe... (is that possible!?), www.baidu.com")
print(tp.tokenize_text(remove_punctuations=True, correct_spelling = False))
```

**method方法说明**

```TextPreprocess.tokenize_text()```

参数 = 默认值（bool）

remove_accentedChar = True, 

remove_httpLinks = True, 

expand_contractionMap = True,

handle_emoji = True, 

convert_word2Number = True,

remove_whiteSpace = True, 

lemmatization = True,

lowercase = True, 

remove_punctuations = True, 

remove_number = True,

remove_specialChars = True, 

expand_slang = True,

remove_stopWords = True,

correct_spelling_byWord = True, 

correct_spelling = False

返回值(使用obj.set_returnTokenStatus(status = bool)设置)：

True:  list[list[str]]

False: list[str]



### Future update(details TBD):

1. ~~可能需要添加另一组空格检验， 颜文字可能导致处理后多余的空格~~
2. 有时候会出现落单的单词，比如“s”,怀疑是缩略词展开字典没有涵盖部分的遗产
3. ~~即使允许punctuation出现在token中，也应该要限制最多出现的次数~~
4. 自定义的pipeline，代码重写提高灵活性和复用性
5. 多语言语言模型嵌入



## 实例化document preprocess

```Python
# 随意找一个iterable的数据对象
doc = train["text"].iloc[:2000]

# 实例化DocPreprocess， 他是TextPreprocess的一个subclass
# 默认参数 model = "en_core_web_sm"
dp = DocPreprocess(doc)

dp.set_returnTokenStatus(False)
# 一样的字典设置
dp.auto_setUp(deselectWordList = ["no","not"],
              contractionMap = CONTRACTION_MAP,
              slangDict = SLANG_DICT)

# 和```TextPreprocess.tokenize_text()```参数相同
docTokenized, errorLog= dp.run_docPreprocessing()
```



### Future update(details TBD):

1. 可能会添加并行，目前速度尚可



## demo

```Python
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")
data.dropna(inplace = True)

#%% pre-process text -i
from DocPreprocess import DocPreprocess
from ContractionMap import ContractionMap

cp = ContractionMap()
cp.dual_contractionMap()
cp.add_newContractionItems({"im":"i am", "Im":"I am", "Hes":"He is", "hes": "he is"})
CONTRACTION_MAP = cp.get_contractionMap()

slangText = pd.read_table("slangDict.txt", delimiter="=", header=None)
SLANG_DICT = {slangText.iloc[i][0]:slangText.iloc[i][1] for i in slangText.index}
SLANG_DICT.update({"R":"Are"})

#%% pre-process text -ii
doc = data["text"].iloc[:100]
dp = DocPreprocess(doc)
dp.set_returnTokenStatus(False)
dp.auto_setUp(deselectWordList = ["no","not"],
              contractionMap = CONTRACTION_MAP,
              slangDict = SLANG_DICT,
              maximumPunctRepeatition = 1)
#%% run it
doc_preprocessed, errorLog = dp.run_docPreprocessing(remove_stopWords = True)

#%% long version
fileName = "head100_shortVersion.csv"
aDf = pd.DataFrame(index = range(len(doc_preprocessed)))
aDf["text"] = doc_preprocessed
aDf["sentiment"] = data["sentiment"].iloc[:100]

aDf.to_csv(os.path.join("tinyData", fileName)) 
```

