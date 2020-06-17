# author: Fangwen Li time:
import pandas as pd
import datetime
def sendnews(news_title=[], news_url=[], type=1):

    # author: Fangwen Li time: 2020/06/15
    """
    input:
    new_title: the list of the title of the news
    news_url:  the list of the link of the news
    type: 1 for positive news and 0 for negative news

    """

    from dingtalkchatbot.chatbot import DingtalkChatbot, CardItem


    new_webhook = 'https://oapi.dingtalk.com/robot/send?access_token=fdc8c721927a1d6a640d7050d318afa880558c69bb63888ae6d989db213cc3b1'
    secret = 'SECed9cb888287a1a6462378752c4e3a33a9d638a322e872fb989dc113a3072e83d'  # 创建机器人时钉钉设置页面有提供
    dingding = DingtalkChatbot(new_webhook, secret=secret)
    positive_pic = 'https://ss3.bdstatic.com/70cFv8Sh_Q1YnxGkpoWK1HF6hhy/it/u=3792737622,1884031801&fm=15&gp=0.jpg'
    negative_pic = 'https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1592241074187&di=fd70398ba4da7c7ce27634c1a1b7d8af&imgtype=0&src=http%3A%2F%2Fimg.mp.itc.cn%2Fupload%2F20170615%2F69f8cc42c3514842bce29182ef5beeb7_th.jpg'
    dingding.send_text(msg='现在推送今天新闻！')

    if type == 1:
        dingding.send_text(msg='以下是积极类型新闻:')
        cards = []
        for title, url in zip(news_title, news_url):
            card = CardItem(title=title, url=url, pic_url=positive_pic)
            cards = cards + [card]
        dingding.send_feed_card(cards)

    if type == 2:
        dingding.send_text(msg='以下是消极类型新闻:')
        cards = []
        for title, url in zip(news_title, news_url):
            card = CardItem(title=title, url=url, pic_url=negative_pic)
            cards = cards + [card]
        dingding.send_feed_card(cards)

if __name__ == '__main__':
    datetime.date.today()
    today = datetime.date.today()
    filename = 'news_'+str(today.year) + str(today.month) + str(today.day)+'.csv'
    data = pd.read_csv(filename)
    positive_title = data[data.classification==1].title
    positive_url = data[data.classification==1].url
    negative_title = data[data.classification==2].title
    negative_url = data[data.classification==2].url
    sendnews(news_title=positive_title, news_url=positive_url, type=1)
    sendnews(news_title=negative_title, news_url=negative_url, type=2)