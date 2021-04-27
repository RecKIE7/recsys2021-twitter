import pymysql

db = pymysql.connect(
    user='reckie', 
    passwd='smkie0409', 
    host='203.153.147.114', 
    db='twitter', 
    charset='utf8'
)

cursor = db.cursor(pymysql.cursors.DictCursor)