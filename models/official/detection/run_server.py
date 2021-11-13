import json
import os

from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
from flask import Flask, request

from dataloader import mode_keys
from projects.fashionpedia.configs import factory as config_factory
from projects.fashionpedia.modeling import factory as model_factory
from utils import input_utils


SERVER_HOST = os.environ['SERVER_HOST']
SERVER_PORT = os.environ['SERVER_PORT']
app = Flask(__name__)


IMAGE_SIZE = [640, 640]
MODEL = 'attribute_mask_rcnn'
CHECKPOINT_PATH = os.environ['CHECKPOINT_PATH']
IMAGE_FILES_PATH = os.environ['IMAGE_FILES_PATH']
POSSIBLE_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png']
CATEGORY_THRESHOLD = 0.85
ATTRIBUTE_THRESHOLDS = [0.159252503, 0.676816745, 0.77634833, 0.557378843, 0.318503039, 0.895786232, 0.129439869,
                        0.119439869, 0.318503039, 0.298596722, 0.099533552, 1.967e-06, 0.318503039, 1.967e-06,
                        1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06,
                        1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 0.009908284,
                        1.967e-06, 0.0935598866, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06,
                        1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06, 1.967e-06]


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


@app.route('/batch_predict', methods=['POST'])
def gas_batch_predict():
    try:
        request_json = request.get_json()

        # get the model class and param
        params = config_factory.config_generator(MODEL)
        model = model_factory.model_generator(params)

        with tf.Graph().as_default():
            image_input = tf.placeholder(shape=(), dtype=tf.string)
            image = tf.io.decode_image(image_input, channels=3)
            image.set_shape([None, None, 3])

            image = input_utils.normalize_image(image)
            image, image_info = input_utils.resize_and_crop_image(
                image,
                IMAGE_SIZE,
                IMAGE_SIZE,
                aug_scale_min=1.0,
                aug_scale_max=1.0)
            image.set_shape([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

            # batching.
            images = tf.reshape(image, [1, IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
            images_info = tf.expand_dims(image_info, axis=0)

            # model inference
            outputs = model.build_outputs(
                images, {'image_info': images_info}, mode=mode_keys.PREDICT)

            # Create a saver in order to load the pre-trained checkpoint.
            saver = tf.train.Saver()
            with tf.Session() as sess:
                print(' - Loading the checkpoint...')
                saver.restore(sess, CHECKPOINT_PATH)

                final_results = []
                image_files = [str(p.resolve()) for p in Path(IMAGE_FILES_PATH).glob("**/*") if p.suffix in POSSIBLE_IMAGE_EXTENSIONS]
                for i, image_file in enumerate(image_files):
                    result = {"image_id": image_file.split('/')[-1], "predictions": []}
                    print(' - Processing image %d...' % i)

                    with tf.gfile.GFile(image_file, 'rb') as f:
                        image_bytes = f.read()

                    image = Image.open(image_file)
                    image = image.convert('RGB')  # needed for images with 4 channels.
                    width, height = image.size
                    np_image = (np.array(image.getdata()).reshape(height, width, 3).astype(np.uint8))

                    predictions_np = sess.run(
                        outputs, feed_dict={image_input: image_bytes})
                    num_detections = int(predictions_np['num_detections'][0])
                    np_scores = predictions_np['detection_scores'][0, :num_detections]
                    np_classes = predictions_np['detection_classes'][0, :num_detections]
                    # forcing category id to start from 0(zero)
                    np_classes = np_classes.astype(np.int32) - 1
                    np_attributes = predictions_np['detection_attributes'][0, :num_detections, :]

                    for ind, score in enumerate(np_scores):
                        if score < CATEGORY_THRESHOLD:
                            continue
                        # TODO: send category name or proper index
                        category_id = np_classes[ind]
                        prediction = {"category": category_id, "attributes": []}

                        # TODO: send attribute name or proper index
                        attribute_scores = np_attributes[ind]
                        attribute_indexes = np.where(attribute_scores >= ATTRIBUTE_THRESHOLDS[category_id])
                        prediction['attributes'] = [
                            {"type": "TO BE ADDED", "values": [attr_ind], "scores": [attribute_scores[attr_ind]]}
                            for attr_ind in attribute_indexes]

                        result['predictions'].append(prediction)

                    final_results.append(result)

        return json.dumps({"status": 200, "message": "Success", "data": final_results}, cls=NpEncoder)
    except Exception as exc:
        print(exc)
        return json.dumps({"status": 500, "message": "Exception: {}".format(exc), "data": []})


if __name__ == "__main__":
    app.run(debug=True, host=SERVER_HOST, port=SERVER_PORT)
