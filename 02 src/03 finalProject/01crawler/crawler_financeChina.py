import logging
import datetime
import requests, os
from bs4 import BeautifulSoup
import pandas as pd

class Logger(object):
    def __init__(self,exeFileName=""):
        self.logger=logging.getLogger('')
        self.logger.setLevel(logging.DEBUG)
        format='%(asctime)s - %(levelname)s -%(name)s : %(message)s'
        formatter=logging.Formatter(format)
        streamhandler=logging.StreamHandler()
        streamhandler.setFormatter(formatter)
        self.logger.addHandler(streamhandler)
        log_filename = "log/"+exeFileName+"_"+datetime.datetime.now().strftime("%Y-%m-%d.log")
        filehandler=logging.FileHandler(log_filename, encoding='utf-8')
        filehandler.setFormatter(formatter)
        self.logger.addHandler(filehandler)
    def debug(self, msg):
        self.logger.debug(msg)
    def info(self, msg):
        self.logger.info(msg)
    def warning(self, msg):
        self.logger.warning(msg)
    def error(self, msg):
        self.logger.error(msg)
    def critical(self, msg):
        self.logger.critical(msg)
    def log(self, level, msg):
        self.logger.log(level, msg)
    def setLevel(self, level):
        self.logger.setLevel(level)
    def disable(self):
        logging.disable(50)


def get_link_content(link):
    # connection error
    try:
        link_res=requests.get(link)
    except Exception as e:
        logger.warning(str(e))
        logger.warning("page not found")
        return(None)

    link_res.encoding=('utf8')
    link_html=link_res.text
    link_soup=BeautifulSoup(link_html, 'html.parser')
    link_artibody=link_soup.find("div",id="fontzoom")

    if not link_artibody:
        logger.debug("link_artibody id: fontzoom fail")
        link_artibody = link_soup.find("div",id="content")
    try:
        link_p=",".join([p.text.strip() for p in link_artibody.findAll("p")])
    except Exception as e:
        # logger.warning( str(e))
        logger.warning("nothing in link_p")
        link_p=""

    link_time_str=""
    try:
        link_span_time2=link_soup.find("span",class_="fl time2")
        link_time_str=link_span_time2.text.strip()[:16]
    except Exception as e:
        logger.debug(str(e)+"the link_time is old version")


    old = False
    # deal with old version web page
    if len(link_time_str)==0:
        try:
            link_span_time2=link_soup.find("span",id="pubtime_baidu")
            link_time_str=link_span_time2.text.strip()[5:]
        except Exception as e:
            logger.warning(str(e)+"use new version to match time still fail")
        old = True
        # print(link_time_str)




    if len(link_time_str)==0:
        logger.warning("nothing in link_time")
        link_time_dt = ""
    else:
        try:
            if not old:
                link_time_dt=datetime.datetime.strptime(link_time_str,"%Y年%m月%d日%H:%M")
            elif old:
                # 2016-04-08 14:51:58
                link_time_dt=datetime.datetime.strptime(link_time_str,"%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning("link_time doesn't match")
            link_time_dt=""
    return(link_p,link_time_dt)


def get_dict_list(ul_list):
    try:
        a_list=ul_list.findAll('a')
    except Exception as e:
        print(str(e))
        a_list=[]
    # dict_list_finance_china=[]
    dict_list=[]
    for a_a in a_list:
        a_title=a_a.text
        logger.info("news title: "+a_title)

        a_link=a_a.get('href')
        logger.info("news link: "+a_link)
        
        a_content,a_time_dt=get_link_content(a_link)

        if a_content and a_time_dt:
            logger.info(a_link+" ...done")
        else:
            logger.info(a_link+" ...fail")

        a_dict={
            "datetime":a_time_dt,
            "title":a_title,
            "link":a_link,
            "content":a_content,
            "source" : "finaceChina"
            }
        dict_list.append(a_dict)
    return(dict_list)



EXEFILENAME="financeChina"
logger=Logger(EXEFILENAME)
logger.info("start running "+EXEFILENAME)

ROOT_LINK="http://finance.china.com.cn/stock/"
sub_link=["ssgs"]#,"ssgs","zqyw"

ssgsLink="http://app.finance.china.com.cn/news/column.php?cname=上市公司&p=176"
zqywLink = "http://app.finance.china.com.cn/news/column.php?cname=证券要闻&p=52"
dpLink = "http://app.finance.china.com.cn/news/column.php?cname=大盘分析&p=18"
# PAGE154
for this_sub_link in sub_link:
    # req_link=ROOT_LINK+this_sub_link
    req_link = ssgsLink
    while(True):
        logger.info("req new page of :"+req_link)
        # connection error
        try:
            res=requests.get(req_link)
        except Exception as e:
            logger.warning(str(e))
            logger.warning("the page not found")
            break

        res.encoding=('utf8')
        html_doc=res.text
        soup = BeautifulSoup(html_doc, 'html.parser')
        ul_list=soup.find("ul",class_="news_list")
        if not ul_list:
            logger.warning("there's no ul_list")

        dict_list=get_dict_list(ul_list)
        all_df=pd.DataFrame(dict_list,columns=["datetime","title","content","link","source"])

        file_path = "news_data/"+EXEFILENAME+"4.csv"
        if (os.path.isfile(file_path)):
            all_df.to_csv(file_path,mode="a",index_label="id", header=False)
            logger.info("append csv page: "+req_link)
        else:
            all_df.to_csv(file_path,mode="a",index_label="id")
            logger.info("new write csv page: "+req_link)
        
        if len(dict_list)>0:
            next_a=soup.find("ul",class_="page").findAll("li")[1].a
            req_link=next_a.get("href")
            if len(req_link)<=0:
                break
            elif req_link.startswith("/news/"):
                req_link="http://app.finance.china.com.cn"+req_link
        else:
            break        

