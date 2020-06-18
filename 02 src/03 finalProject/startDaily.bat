call activate env_crawler
cd D:\AIModel\01 crawler
python crawler_sinaCompany.py
pause

call activate env_tensorflow
cd D:\AIModel\03 bert
python load_model_and_out_csv.py
pause

call activate env_dingding
cd D:\AIModel\04 dingtalk_chatbot
python sendnews.py
pause