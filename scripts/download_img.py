import mysql.connector
import requests
import os
from concurrent.futures import ThreadPoolExecutor

# 图片保存目录
SAVE_DIR = "baoan"
os.makedirs(SAVE_DIR, exist_ok=True)

cnx = mysql.connector.connect(
    host="124.71.17.254",
    port="53308",
    user="jgxl_jmpt",
    password="ynO3+3k0hE=s7",
    database="modeling_platform_zlgl_test_hasdata_816"
)

# 创建游标
cursor = cnx.cursor()

# 执行SQL查询
query = ("SELECT scenePhoto FROM test_aidetect WHERE tag LIKE '%保安%'")
cursor.execute(query)


def download_image(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = url.split('?')[0].split('/')[-1]
            with open(f'baoan/{filename}', 'wb') as f:
                f.write(response.content)
                print(f'Successfully downloaded {filename}')
        else:
            print(f'Failed to download image from {url}')
    except Exception as e:
        print(f'Error downloadingimage from {url}: {e}')


urls = [row[0] for row in cursor.fetchall()]
with ThreadPoolExecutor(max_workers=5) as executor:
    future = {executor.submit(download_image, url): url for url in urls}

# 关闭游标和连接
cursor.close()
cnx.close()
