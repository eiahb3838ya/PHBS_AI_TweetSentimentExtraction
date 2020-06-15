import logging
import datetime
import requests,os
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
        return("","")

    link_res.encoding=('utf8')
    link_html=link_res.text
    link_soup=BeautifulSoup(link_html, 'html.parser')
    link_artibody=link_soup.find("div",class_="edittext")
    try:
        link_p=link_artibody.text.strip()
    except Exception as e:
        logger.warning(str(e))
        logger.warning("nothing in link_p")
        link_p=""

    link_newstitle2 = link_soup.find("div",class_="newstitle2")
    # use better way later to get time
    link_time_str=""
    for string in link_newstitle2.stripped_strings:
        if string[0].isdigit():
            link_time_str=string

    if len(link_time_str)==0:
        logger.warning("nothing in link_time")
        link_time_dt = ""
    else:
        try:
            link_time_dt=datetime.datetime.strptime(link_time_str,"%Y-%m-%d %H:%M:%S")
        except Exception as e:
            logger.warning("link_time doesn't match")
            link_time_dt=""
    return(link_p,link_time_dt)


def get_dict_list(ul_list):
    a_ul=ul_list[0]
    a_list=a_ul.findAll('a')
    dict_list=[]
    for a_a in a_list:
        a_title=a_a.text
        logger.info("news title: "+a_title)

        a_link=a_a.get('href')
        fix_part_link="https://www.lj168.com"
        a_link=fix_part_link+a_link
        logger.info("news link: "+a_link)
        
        a_content,a_time=get_link_content(a_link)

        if a_content and a_time:
            logger.info(a_link+" ...done")
        else:
            logger.info(a_link+" ...fail")

        a_dict={
            "datetime":a_time,
            "title":a_title,
            "link":a_link,
            "content":a_content,
            "source":"lj168"
            }
        dict_list.append(a_dict)
    return(dict_list)



EXEFILENAME="lj168"
logger=Logger(EXEFILENAME)
logger.info("start running "+EXEFILENAME)
ROOT_LINK="https://www.lj168.com/1008/dissort?"
for page_num in range(1,561):
    req_link=ROOT_LINK+"page="+str(page_num)
    logger.info("req new page of :"+req_link)
    # connection error
    try:
        res=requests.get(req_link)
    except Exception:
        logger.warning(Exception)
        logger.warning("the page not found")
        continue

    res.encoding=('utf8')
    html_doc=res.text
    soup = BeautifulSoup(html_doc, 'html.parser')
    ul_list=soup.findAll('ul',class_="newsul01")
    if not ul_list:
        logger.warning("there's no ul_list")

    dict_list=get_dict_list(ul_list)
    all_df=pd.DataFrame(dict_list,columns=["datetime","title","content","link","source"])

    file_path = "news_data/"+EXEFILENAME+".csv"
    if (os.path.isfile(file_path)):
            all_df.to_csv(file_path,mode="a",index_label="id", header=False)
            logger.info("append csv page: "+req_link)
    else:
        all_df.to_csv(file_path,mode="a",index_label="id")
        logger.info("new write csv page: "+req_link)
    # all_df.to_csv("news_data/"+EXEFILENAME+".csv",mode="a")

