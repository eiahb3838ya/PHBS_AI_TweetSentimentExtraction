# Meeting Logs

## Contents

[Apr-29](#Apr-29)

[Apr-30](#Apr-30)

[May-2](#May-2)



## Apr-29

阅读主要参考文献，安装lib： （Python3.7.x）re, nltk, spacy, json, genism

无细节分工



## Apr-30

pre-processing of text

大致的步骤

1. 处理奇怪的空格
2. 分词（tokenization）
3. 拼写矫正
4. 缩写矫正
5. 词源
6. emoji
7. 停止词

在处理分词的时候注意：

(a)专有名词

(b)标点符号（假冒的标点，比如16：20，不用于语义分割）

(c)颜文字

(d)emoji

(e)脏话（\****）

(f)乱码

(g)特殊含义字符(@,#)

(h) URL

# May-2
完成了beta版本的pre-process text,具体性能有待进一步测试，目前使用的模型中，专用名词分割的准确率为85%左右，spelling-check的准确度为70%左右（因此引入了correct_spelling_byWord，作为替代保证所有重复字母连续出现不超过两次，会导致"..."变成".."， "sooooo"变成"soo"）

word embedding模型的分工
word2Vec - Edward : task: 1. run model, 2. 2-d visualization(PCA or t-SNE)
BERT - Robert: task: 1. run model, 2. fine tunning, 3. 2-d visualization(PCA or t-SNE)
