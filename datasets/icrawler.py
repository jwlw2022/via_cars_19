from icrawler.builtin import GoogleImageCrawler
import shutil
import os

label = 'Porsche 911 Speedster 2019'
base_dir = os.path.dirname(os.path.realpath('__filename__'))
goog_path = os.path.join(base_dir,label)

if os.path.exists(goog_path):
    shutil.rmtree(goog_path)
os.mkdir(goog_path)

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': '217_new'})
filters = dict(
    size='large',
    #color='orange',
    license='None')
    #date=((2017, 1, 1), (2017, 11, 30)))

google_crawler.crawl(keyword=label, filters=filters, max_num=300, file_idx_offset=0)
