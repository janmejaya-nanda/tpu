# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=line-too-long
r"""A stand-alone binary to run model inference and visualize results.

It currently only supports model of type `retinanet` and `mask_rcnn`. It only
supports running on CPU/GPU with batch size 1.
"""
# pylint: enable=line-too-long

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import base64
import csv
import io
import json

from absl import flags
from absl import logging

import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_api
import tensorflow.compat.v1 as tf
import statistics

from dataloader import mode_keys
from projects.fashionpedia.configs import factory as config_factory
from projects.fashionpedia.modeling import factory as model_factory
from utils import box_utils
from utils import input_utils
from utils import mask_utils
from utils.object_detection import visualization_utils
from hyperparameters import params_dict


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'model', 'attribute_mask_rcnn', 'Support `attribute_mask_rcnn`.')
flags.DEFINE_integer('image_size', 640, 'The image size.')
flags.DEFINE_string(
    'checkpoint_path', '', 'The path to the checkpoint file.')
flags.DEFINE_string(
    'config_file', '', 'The config file template.')
flags.DEFINE_string(
    'params_override', '', 'The YAML file/string that specifies the parameters '
    'override in addition to the `config_file`.')
flags.DEFINE_string(
    'label_map_file', '',
    'The label map file. See --label_map_format for the definition.')
flags.DEFINE_string(
    'label_map_format', 'csv',
    'The format of the label map file. Currently only support `csv` where the '
    'format of each row is: `id:name`.')
flags.DEFINE_string(
    'image_file_pattern', '',
    'The glob that specifies the image file pattern.')
flags.DEFINE_string(
    'output_html', '/tmp/test.html',
    'The output HTML file that includes images with rendered detections.')
flags.DEFINE_string(
    'output_file', '/tmp/res.npy',
    'The output npy file that includes model output.')
flags.DEFINE_integer(
    'max_boxes_to_draw', 10, 'The maximum number of boxes to draw.')
flags.DEFINE_float(
    'min_score_threshold', 0.05,
    'The minimum score thresholds in order to draw boxes.')
flags.DEFINE_string('output_coco', '/tmp/out-coco.json', "Whether to save output in COCO format.")
flags.DEFINE_string('attribute_json', None, "Json having attributes and ID mapping")
flags.DEFINE_string('result_csv_path', None, "path to save result in CSV")
flags.DEFINE_string('meta_file_path', None, "path to meta file")
flags.DEFINE_string('weights_path', None, "weights path")
flags.DEFINE_string('attribute_threshold_json', None, "A JSON containing threshold for each category and attribute "
                                                      "combination")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def main(unused_argv):
  del unused_argv
  # Load the label map.
  print(' - Loading the label map...')
  label_map_dict = {}
  if FLAGS.label_map_format == 'csv':
    with tf.gfile.Open(FLAGS.label_map_file, 'r') as csv_file:
      reader = csv.reader(csv_file, delimiter=':')
      for row in reader:
        if len(row) != 2:
          raise ValueError('Each row of the csv label map file must be in '
                           '`id:name` format.')
        id_index = int(row[0])
        name = row[1]
        label_map_dict[id_index] = {
            'id': id_index,
            'name': name,
        }
  else:
    raise ValueError(
        'Unsupported label map format: {}.'.format(FLAGS.label_mape_format))

  params = config_factory.config_generator(FLAGS.model)
  if FLAGS.config_file:
    params = params_dict.override_params_dict(
        params, FLAGS.config_file, is_strict=True)
  params = params_dict.override_params_dict(
      params, FLAGS.params_override, is_strict=True)
  params.override({
      'architecture': {
          'use_bfloat16': False,  # The inference runs on CPU/GPU.
      },
  }, is_strict=True)
  params.validate()
  params.lock()

  model = model_factory.model_generator(params)

  with tf.Graph().as_default():
    image_input = tf.placeholder(shape=(), dtype=tf.string)
    image = tf.io.decode_image(image_input, channels=3)
    image.set_shape([None, None, 3])

    image = input_utils.normalize_image(image)
    image_size = [FLAGS.image_size, FLAGS.image_size]
    image, image_info = input_utils.resize_and_crop_image(
        image,
        image_size,
        image_size,
        aug_scale_min=1.0,
        aug_scale_max=1.0)
    image.set_shape([image_size[0], image_size[1], 3])

    # batching.
    images = tf.reshape(image, [1, image_size[0], image_size[1], 3])
    images_info = tf.expand_dims(image_info, axis=0)

    # model inference
    outputs = model.build_outputs(
        images, {'image_info': images_info}, mode=mode_keys.PREDICT)

    outputs['detection_boxes'] = (
        outputs['detection_boxes'] / tf.tile(images_info[:, 2:3, :], [1, 1, 2]))

    predictions = outputs

    # Create a saver in order to load the pre-trained checkpoint.
    saver = tf.train.Saver()

    image_with_detections_list = []
    with tf.Session() as sess:
      print(' - Loading the checkpoint...')
      saver.restore(sess, FLAGS.checkpoint_path)

      res = []
      image_files = tf.gfile.Glob(FLAGS.image_file_pattern)
      for i, image_file in enumerate(image_files):
        print(' - Processing image %d...' % i)

        with tf.gfile.GFile(image_file, 'rb') as f:
          image_bytes = f.read()

        image = Image.open(image_file)
        image = image.convert('RGB')  # needed for images with 4 channels.
        width, height = image.size
        np_image = (np.array(image.getdata())
                    .reshape(height, width, 3).astype(np.uint8))

        predictions_np = sess.run(
            predictions, feed_dict={image_input: image_bytes})

        num_detections = int(predictions_np['num_detections'][0])
        np_boxes = predictions_np['detection_boxes'][0, :num_detections]
        np_scores = predictions_np['detection_scores'][0, :num_detections]
        np_classes = predictions_np['detection_classes'][0, :num_detections]
        np_classes = np_classes.astype(np.int32)
        np_attributes = predictions_np['detection_attributes'][
            0, :num_detections, :]
        np_masks = None
        if 'detection_masks' in predictions_np:
          instance_masks = predictions_np['detection_masks'][0, :num_detections]
          np_masks = mask_utils.paste_instance_masks(
              instance_masks, box_utils.yxyx_to_xywh(np_boxes), height, width)
          encoded_masks = [
              mask_api.encode(np.asfortranarray(np_mask))
              for np_mask in list(np_masks)]

        res.append({
            'image_file': image_file,
            'boxes': np_boxes,
            'classes': np_classes,
            'scores': np_scores,
            'attributes': np_attributes,
            'masks': encoded_masks,
            'attributes_prob': sess.run(tf.nn.softmax(np_attributes, axis=1))
        })

        image_with_detections = (
            visualization_utils.visualize_boxes_and_labels_on_image_array(
                np_image,
                np_boxes,
                np_classes,
                np_scores,
                label_map_dict,
                instance_masks=np_masks,
                use_normalized_coordinates=False,
                max_boxes_to_draw=FLAGS.max_boxes_to_draw,
                min_score_thresh=FLAGS.min_score_threshold))
        image_with_detections_list.append(image_with_detections)

  # preparing result in CSV
  if FLAGS.result_csv_path:
    if not (FLAGS.attribute_json and FLAGS.attribute_threshold_json):
      raise Exception("Missing attribute json mapping")
    with open(FLAGS.attribute_json) as f:
      data = json.load(f)
      attributes_map = dict([(attribute['id'], attribute['name']) for attribute in data['attributes']])
      attribute_ids = np.array([i[0] for i in sorted(attributes_map.items(), key=lambda x: x[0])])
    with open(FLAGS.attribute_threshold_json) as f:
      attr_data = json.load(f)
      attribute_thresholds = dict([(int(i), round(np.mean(v))) for i, v in attr_data.items()])

    csv_data = []
    for i in res:
      for indx, attribute_score in enumerate(i['attributes_prob']):
        if i['scores'][indx] < 0.8:
          continue
        
        category_id = i['classes'][indx]
        top_k = attribute_thresholds[category_id]
        if top_k:
          ind = np.argpartition(attribute_score, len(attribute_score) - top_k)[-top_k:]
          predicted_attribute_ids = attribute_ids[ind]
          attribute_values = [attributes_map[i] for i in predicted_attribute_ids]
        else:
          ind, predicted_attribute_ids, attribute_values = [], [], []

        # if attribute_value in required_attributes:
        csv_data.append([i['image_file'].split('/')[-1], label_map_dict[category_id]['name'], ','.join(attribute_values), i['boxes'][indx], i['masks'][indx]])
    csv_columns = ['Images file', 'Category value', 'Attribute value', 'Bounding Boxes', "Mask"]
    df = pd.DataFrame(data=csv_data, columns=csv_columns)
    df.to_csv(FLAGS.result_csv_path)

  print(' - Saving the outputs...')
  formatted_image_with_detections_list = [
      Image.fromarray(image.astype(np.uint8))
      for image in image_with_detections_list]
  html_str = '<html>'
  image_strs = []
  for formatted_image in formatted_image_with_detections_list:
    with io.BytesIO() as stream:
      formatted_image.save(stream, format='JPEG')
      data_uri = base64.b64encode(stream.getvalue()).decode('utf-8')
    image_strs.append(
        '<img src="data:image/jpeg;base64,{}", height=800>'
        .format(data_uri))
  images_str = ' '.join(image_strs)
  html_str += images_str
  html_str += '</html>'
  with tf.gfile.GFile(FLAGS.output_html, 'w') as f:
    f.write(html_str)
  np.save(FLAGS.output_file, res)

  # saving result in COCO format
  if FLAGS.output_coco:
    if not (FLAGS.attribute_json and FLAGS.attribute_threshold_json):
      raise Exception("Missing attribute json mapping")
    with open(FLAGS.attribute_json) as f:
      data = json.load(f)
      attributes_map = dict([(attribute['id'], attribute['name']) for attribute in data['attributes']])
      attribute_ids = np.array([i[0] for i in sorted(attributes_map.items(), key=lambda x: x[0])])
    with open(FLAGS.attribute_threshold_json) as f:
      attr_data = json.load(f)
      attribute_thresholds = dict([(int(i), round(np.mean(v))) for i, v in attr_data.items()])

    print("saving result in COCO format to: {}".format(FLAGS.output_coco))
    print("$"*40)
    coco_result = []
    for i in res:
      for box, category_id, attribute_score, score in zip(i['boxes'], i['classes'], i['attributes_prob'], i['scores']):
        if score < 0.8:
          continue

        top_k = attribute_thresholds[category_id]
        print("len(attribute_score)", len(attribute_score))
        print("top_k", top_k)

        if top_k:
          ind = np.argpartition(attribute_score, len(attribute_score) - top_k)[-top_k:]
          predicted_attribute_ids = attribute_ids[ind]
          attribute_values = [attributes_map[i] for i in predicted_attribute_ids]
        else:
          ind, predicted_attribute_ids, attribute_values = [], [], []

        print("category_id", category_id)
        print("class Name: ", label_map_dict[category_id]['name'])
        print("allowed n attributes", top_k)
        print("attribute_score", attribute_score)
        print("index", ind)
        print("Attribute IDs: ", predicted_attribute_ids)
        print("Attribute Name: ", attribute_values)
  
        coco_result.append({
          "image_id": i['image_file'].split('/')[-1].replace('.jpg', ''), 
          "category_id": category_id, 
          "attribute_ids": predicted_attribute_ids,
          "bbox": i['boxes'], 
          "score": i['scores'],
        })
    with open(FLAGS.output_coco, 'w') as f:
        json.dump(coco_result, f, cls=NpEncoder)
  print("$"*40)


if __name__ == '__main__':
  flags.mark_flag_as_required('model')
  flags.mark_flag_as_required('checkpoint_path')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('image_file_pattern')
  flags.mark_flag_as_required('output_html')
  logging.set_verbosity(logging.INFO)
  tf.app.run(main)
