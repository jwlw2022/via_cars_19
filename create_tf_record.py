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
r"""Convert raw Stanford Cars dataset to TFRecord for object_detection.

Example usage:
    python create_stanford_cars_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_train.tfrecord \
        --set=train \
        --label_map_path=stanford_cars_labels_map.pbtxt

    python create_stanford_cars_tf_record.py \
        --data_dir=/data/StanfordCars \
        --output_path=stanford_cars_test.tfrecord \
        --set=test \
        --label_map_path=stanford_cars_labels_map.pbtxt
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import hashlib
import io
import logging
import csv

import PIL.Image

import numpy as np

import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


flags = tf.app.flags

flags.DEFINE_string('data_dir','','Root directory to Stanford Cars dataset. (car_ims is a subfolder)')
flags.DEFINE_string('output_path','new_stanford_cars.tfrecord','Path to output TFRecord.')
flags.DEFINE_string('label_map_path','new_stanford_cars_label_map.pbtxt','Path to label map proto.')
flags.DEFINE_string('set','merged','Convert training set, test set, or merged set.')
flags.DEFINE_string('csv','','Converted CSV labels file')

FLAGS = flags.FLAGS

SETS = ['train', 'test', 'merged']

def dict_to_tf_example(annotation, dataset_directory, label_map_dict):
  im_path = str(annotation['relative_im_path'])
  cls = int(annotation['class'])
  x1 = int(annotation['bbox_x1'])
  y1 = int(annotation['bbox_y1'])
  x2 = int(annotation['bbox_x2'])
  y2 = int(annotation['bbox_y2'])

  # read image
  full_img_path = os.path.join(dataset_directory, im_path)

  # read in the image and make a thumbnail of it
  max_size = 500, 500
  big_image = PIL.Image.open(full_img_path)
  width,height = big_image.size
  big_image.thumbnail(max_size, PIL.Image.ANTIALIAS)
  full_thumbnail_path = os.path.splitext(full_img_path)[0] + '_thumbnail.jpg'
  big_image.save(full_thumbnail_path)

  with tf.gfile.GFile(full_thumbnail_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)

  xmin = []
  xmax = []
  ymin = []
  ymax = []

  # calculate box using original image coordinates
  xmin.append(max(0, x1 / width))
  xmax.append(min(1.0, x2 / width))
  ymin.append(max(0, y1 / height))
  ymax.append(min(1.0, y2 / height))

  # set width and height to thumbnail size for tfrecord ingest
  width,height = image.size

  classes = []
  classes_text = []

  label=''
  for name, val in label_map_dict.items():
    if val == cls: 
      label = name
      break

  classes_text.append(label.encode('utf8'))
  classes.append(label_map_dict[label])
  
  example = tf.train.Example(features=tf.train.Features(feature={
	'image/height': dataset_util.int64_feature(height),
	'image/width': dataset_util.int64_feature(width),
	'image/filename': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/source_id': dataset_util.bytes_feature(full_img_path.encode('utf8')),
	'image/encoded': dataset_util.bytes_feature(encoded_jpg),
	'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
	'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
	'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
	'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
	'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
	'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
	'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return example 

def main(_):

  if FLAGS.set not in SETS:
    raise ValueError('set must be in : {}'.formats(SETS))

  train = FLAGS.set
  data_dir = FLAGS.data_dir
  csv_file = FLAGS.csv

  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  label_map_dict = label_map_util.get_label_map_dict(FLAGS.label_map_path)

  with open(csv_file) as f:
    csv_reader = csv.DictReader(f)
    for row in csv_reader:
      test = int(row['test'])
      if test:
        testset = 'test'
      else:
        testset = 'train'

      if train == 'merged' or train == testset:
        tf_example = dict_to_tf_example(row, data_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
