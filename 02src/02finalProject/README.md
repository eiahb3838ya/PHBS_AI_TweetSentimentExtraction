## 基于钉钉机器人的财经新闻订阅服务

#### Background

这个项目会通过python的爬虫从新浪财经上下载实时的每日上市公司公告的数据。用ChinaScope提供的已经清洗完的正向、负向、中性情感文本用Bert模型进行训练，sample data的数量是19w条。后端部分是用训练完的Bert模型来对实时爬下来的新浪财经公司公告数据进行情感分类。前端部分是用DingChatBot推送正向和负向情感的前5条新闻。

#### Install Packages In Python

1. crawler：bs4,pandas

2. bert：tensorflow1.14.0-gpu, tensorflow_hub, bert

3. dingding：dingtalkchatbot 1.5.0

#### Usage

1. 将整个文件夹的03 finalProject clone到本地
2. 先运行 01 crawler下的[crawler_sinaCompany.py](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/01crawler/crawler_sinaCompany.py)，实时从新浪财经上download上市公司新闻公告，数据会存入02 news_data
3. 获取bert模型情感分类的训练数据，数据格式类似05 word_segments下的[sample1.json](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/05word_segments/sample1.json)，acturally we train model use the ChinaScope's data which has sentiment tag(012). The bigger the dataset, the better the later performance. 
4. 运行 03 bert下的[AI_bert_for_final.py](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/03bert/AI_bert_for_final.py)，会训练模型，如果是19w条sample data，this project actually spent about 3 hours to train bert model. This is related to your pc and the parameters set in the bert model. 这个.py的output结果会生成训练好的model。
5. 运行 03 bert下的[load_model_and_out_csv.py](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/03bert/load_model_and_out_csv.py)，进行了上市公司新闻公告的情感分类，output的结果会存入02 news_data
6. 运行 04 dingtalk_chatbot下的[sendnews.py](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/04dingtalk_chatbot/sendnews.py)，这里需要设置钉钉机器人的webhook地址和secret，输出结果是正向和负向上市公司新闻公告中前5的新闻，这会直接print到钉钉群中。
7. 更简单的办法是：直接双击03 finalProject 下的 [startDaily.bat](https://github.com/eiahb3838ya/PHBS_AI_TweetSentimentExtraction/blob/master/02src/03finalProject/startDaily.bat)即可通过batch文件运行，里面的路径需要自己cd修改一下。

#### Environment

Recommand to create three new environment seperately for **crawler/bert/dingchatbot**.

这个项目需要在本地部署GPU版本的tensorflow进行bert模型的训练。

下给出Window10 操作系统下安装GPU版本的tensorflow的Install steps。

1. 操作系统:Window10
2. GPU:NVIDIA GeForce GTX 1070 Ti
3. Create New Environment in Anaconda: env_tensorflow 
4. Python:3.7
5. 安装Visual Studio 2017
6. CUDA:安装CUDA 10.0
7. CUDNN:安装CUDNN 7.6
8. TensorFlow Version(GPU):1.14.0
9. Bert-TensorFlow Version:1.0.1

#### Contributors

这项目是由PHBS.2020.M4.AI.Group1的小组同学。Thanks for their work.

#### Input and Output

1. Input：爬虫爬下来的上市公司新闻公告数据
2. Output：上市公司新闻公告数据新闻情感分类完的结果并在钉钉群里推送

#### License

[MIT License](https://github.com/RichardLitt/standard-readme/blob/master/LICENSE) 