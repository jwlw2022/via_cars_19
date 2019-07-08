# Copyright 2019 VIA Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Predict vehicle Bounding box and class on an image.

Example usage:
  python predict_image.py \
    --image <image_path> \
    --labels stanford_cars_label_map.pbtxt \
    --model frozen_inference_graph.pb
"""
import os
import csv
import random
import sys
import six.moves.urllib as urllib
import tarfile
import zipfile
import subprocess

from io import StringIO
from PIL import Image

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import ops as utils_ops
from object_detection.utils import visualization_utils as vis_util

flags = tf.app.flags
flags.DEFINE_string('model','frozen_inf_graph.pb','Frozen graph file')
flags.DEFINE_string('labels','2019_cars_label_map.pbtxt','pbtxt labels file')
#flags.DEFINE_string('image','','Image to run prediction on')
FLAGS = flags.FLAGS

NUM_CLASSES = 30

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def run_inference_for_single_image(image, graph):
  # TODO load_image_into_numpy_array
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
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]

  return output_dict

def file_list(mypath):
  return [f for f in os.listdir(mypath) if os.path.isfile(os.path.join(mypath, f))]

files = file_list('datasets/test')
TEST_IM_PATHS = [ os.path.join('datasets/test', '{}'.format(i)) for i in files ]

def main(_):
  answers = []
  guesses = []

  for i in TEST_IM_PATHS[:50]:
    #answers.append(i)
    #print(i)

    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(FLAGS.model, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
  
    label_map = label_map_util.load_labelmap(FLAGS.labels)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
  
    image = Image.open(i)
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
    output_dict = run_inference_for_single_image(image_np, detection_graph)

    print(output_dict['detection_scores'][0], output_dict['detection_classes'][0])
    class_pred = str(output_dict['detection_classes'][0])
    if len(class_pred) == 1:
      class_pred = '0' + class_pred

    guess = category_index[output_dict['detection_classes'][0]]['name']

    print(guess)
    guesses.append(guess)
    print(i)
    answers.append(i)
    
    with open('test_faster_rcnn_resnet101.csv', mode='a') as results:
      results_writer = csv.writer(results, delimiter=',', quoting=csv.QUOTE_MINIMAL)
      if i[14:16] == class_pred: 
        results_writer.writerow([i[14:-4], guess, 1])
      else:
        results_writer.writerow([i[14:-4], guess, 0])

    #with open('test_model4.csv', 'rb') as f:
    #    data = list(csv.reader(f))

    #writer = csv.writer(open("test_model4.csv", 'w'))
    #writer.writerow([i, guess])

  # Visualization of the results of a detection.
  '''
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks'),
      use_normalized_coordinates=True,
      line_thickness=8)

  myim = Image.fromarray(image_np)
  myim_basename = os.path.basename(FLAGS.image)
  myim.save(os.path.join('results6/', myim_basename)) 
  '''
if __name__ == '__main__':
  tf.app.run()
  #np.savetxt('test_model4.csv', (answers, guesses), delimiter=',')
