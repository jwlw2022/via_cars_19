import os
import predict_image
import subprocess

def file_list(mypath):
    return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

files = file_list('datasets/test')
TEST_IM_PATHS = [ os.path.join('datasets/test', '{}'.format(i)) for i in files ]

for i in TEST_IM_PATHS:
    print(i)
    os.system('python3 predict_image.py --model frozen_inf_graph.pb --labels 2019_cars_label_map.pbtxt --image {}'.format(i))
    #print(python3 predict_image.py --model frozen_inference_graph.pb --labels 2019_cars_label_map.pbtxt --image datasets/test/
