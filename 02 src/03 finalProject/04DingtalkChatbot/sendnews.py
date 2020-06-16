# author: Fangwen Li time:
def sendnews(news_title=[], news_url=[], type=1):

    # author: Fangwen Li time: 2020/06/15
    """
    input:
    new_title: the list of the title of the news
    news_url:  the list of the link of the news
    type: 1 for positive news and 0 for negative news

    """
    from dingtalkchatbot.chatbot import DingtalkChatbot, CardItem
    new_webhook = 'https://oapi.dingtalk.com/robot/send?access_token=bb1fda685c28f7f70c36f114715488b70d9e4e427e9690e3702439958e7f7d0c'
    secret = 'SEC750b06eda22414656e9983c44538812c0d970a9a4d8c43cf8b4e25c9b1ee4963'  # 创建机器人时钉钉设置页面有提供
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

    if type == 0:
        dingding.send_text(msg='以下是消极类型新闻:')
        cards = []
        for title, url in zip(news_title, news_url):
            card = CardItem(title=title, url=url, pic_url=negative_pic)
            cards = cards + [card]
        dingding.send_feed_card(cards)

if __name__ == '__main__':
    positive_title =["已经上岸的基金投资界萌新，教你傻瓜式A类C类费用如何计算？","创业板注册制来了 下周一开始受理！涨跌幅限制调整至20%","彻底火了！城管打电话喊商贩摆摊，各地加码地摊经济！这些公司努力“摊”上事"]
    positive_url = ["https://xueqiu.com/5504896272/151471487","https://xueqiu.com/5124430882/151500672","https://xueqiu.com/5124430882/150721515"]
    sendnews(news_title=positive_title,news_url=positive_url,type=0)