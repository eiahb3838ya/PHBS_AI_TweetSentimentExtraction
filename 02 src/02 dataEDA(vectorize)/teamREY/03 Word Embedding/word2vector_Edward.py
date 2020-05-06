from gensim.models.word2vec import Word2Vec
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df =  pd.read_csv('head100_longVersion.csv')
text_list = df['text']
# print(text_list)

list_of_tokens = []
for i in range(1,len(text_list)):
    text_record = text_list[i]
#     print(text_record)
    words=text_record.split(' ')
#     print(words)
    list_of_tokens.append(words)

print('num of sentences:', len(list_of_tokens))

model = Word2Vec(size=20, workers=5,sg=1)
model.build_vocab(list_of_tokens)
model.train(list_of_tokens,total_examples = model.corpus_count,epochs = model.epochs)
# model.save('gensim_w2v_model')            # 保存a模型 
# new_model = Word2Vec.load('gensim_w2v_model') # 调用模型

print('all words in dic:',model.wv.index2word)    #获得所有的词汇
print('the len of dic: ',len(model.wv.index2word))    #获得所有的词汇
words_vector_matrix = []
for word in model.wv.index2word:
#     print(word,model.wv[word])     #获得词汇及其对应的向量
    words_vector_matrix.append(model.wv[word])

# # 输出相近的词语和概率
# sim_words = model.most_similar(positive=['like'])
# for word,similarity in sim_words:
#     print(word,similarity)
    
# #计算余弦距离最接近的20个词
# for i in model.most_similar("like", topn=20):
#     print(i[0],i[1])
    
# # 输出的词向量
# print(model['like'])

# #计算两个词之间的余弦距离
# y2=model.similarity("like", "love") 
# print(y2)

estimator = PCA(n_components=2)
# estimator.fit(x)
pca_vector=estimator.fit_transform(words_vector_matrix)

xValue = [ vector[0] for vector in pca_vector]
yValue = [ vector[1] for vector in pca_vector]

plt.title('Demo')
plt.xlabel('pca value-1')
plt.ylabel('pca value-2')

# s：点的大小/粗细 标量或array_like 默认是 rcParams['lines.markersize'] ** 2
plt.scatter(xValue, yValue, s=20, c="#ff1212", marker='o')
# plt.scatter(train_x, train_y_1, c='red', marker='v' )
# plt.scatter(train_x, train_y_2, c='blue', marker='o' )
plt.legend(['red'])
# plt.legend(["red","Blue"])
plt.show()

