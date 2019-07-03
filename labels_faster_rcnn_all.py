import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("../models/research/object_detection") #REPLACE
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# This is needed to display the images.
#%matplotlib inline

from utils import label_map_util

from utils import visualization_utils as vis_util

# What model to download.
MODEL_NAME = 'faster_rcnn_nas_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '2019_cars_label_map.pbtxt' #REPLACE

#opener = urllib.request.URLopener()
#opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
tar_file = tarfile.open(MODEL_FILE)
for file in tar_file.getmembers():
  file_name = os.path.basename(file.name)
  if 'frozen_inference_graph.pb' in file_name:
    tar_file.extract(file, os.getcwd())

print('working')

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print('working')

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

#Obtain list of all files in image directory
from os import listdir
from os.path import isfile, join
def file_list(mypath):
  return [f for f in listdir(mypath) if isfile(join(mypath, f))]
  
# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.

for curr_dir in range(199, 227):
  PATH_TO_TEST_IMAGES_DIR = 'datasets/{}'.format(curr_dir) #REPLACE
  files = file_list(PATH_TO_TEST_IMAGES_DIR)
  TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(i)) for i in files ]   
  #TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, '{}'.format(i)) for i in range(1, 83) ]    
  #REPLACE

  # Size, in inches, of the output images.
  IMAGE_SIZE = (12, 8)

  def run_inference_for_single_image(image, graph):
    with graph.as_default():
      with tf.Session() as sess:
        # Get handles to input and output tensors
        ops = tf.get_default_graph().get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
          tensor_name = key + ':0'
          if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)
        if 'detection_masks' in tensor_dict:
          # The following processing is only for single image
          detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
          detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
         # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
          real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
          detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
          detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
          detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              detection_masks, detection_boxes, image.shape[1], image.shape[2])
          detection_masks_reframed = tf.cast(
              tf.greater(detection_masks_reframed, 0.5), tf.uint8)
          # Follow the convention by adding back the batch dimension
          tensor_dict['detection_masks'] = tf.expand_dims(
              detection_masks_reframed, 0)
        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # Run inference
        output_dict = sess.run(tensor_dict,
                               feed_dict={image_tensor: image})

        # all outputs are float32 numpy arrays, so convert types as appropriate
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
          output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

  for image_path in TEST_IMAGE_PATHS:
    print('working')
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)
    #print(output_dict)
  
    # Visualization of the results of a detection.
    #vis_util.visualize_boxes_and_labels_on_image_array(
    #    image_np,
    #    output_dict['detection_boxes'],
    #    output_dict['detection_classes'],
    #    output_dict['detection_scores'],
    #    category_index,
    #    instance_masks=output_dict.get('detection_masks'),
    #    use_normalized_coordinates=True,
    #    line_thickness=8)
    #plt.figure(figsize=IMAGE_SIZE)
    #plt.imshow(image_np)
    #fig_path = image_path.replace("{}/".format(curr_dir), "{}_boxes/".format(curr_dir))
    #plt.savefig(fig_path)

    xml_name = image_path.replace('.jpg', '.csv')
    xml_name = xml_name.replace("{}/".format(curr_dir), "{}_csv2/".format(curr_dir))
    bbox_list = []
    for item in output_dict['detection_boxes'][0]:
      bbox_list.append(item)
    bbox_list[0] = int(round(bbox_list[0]*image.size[0]))
    bbox_list[1] = int(round(bbox_list[1]*image.size[1]))
    bbox_list[2] = int(round(bbox_list[2]*image.size[0]))
    bbox_list[3] = int(round(bbox_list[3]*image.size[1]))
    f= open(xml_name,"w+")
    f.write(str(image_path))
    f.write(",{},".format(curr_dir - 196))
    for i in bbox_list:
      f.write(str(i))
      f.write(",")
