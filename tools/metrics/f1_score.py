"""
Given result(JSON) in COCO format and validation(JSON), below code finds attribute F1 score
"""
import gc
import json

import numpy as np
from absl import app, flags
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

FLAGS = flags.FLAGS

flags.DEFINE_string('result_json', None, 'prediction JSON (in COCO format) file path', required=False)
flags.DEFINE_string('validation_json', None, 'validation JSON file path', required=False)


def main(argv):
    with open(FLAGS.result_json) as f:
        result_json = json.load(f)
    with open(FLAGS.validation_json) as f:
        validation_json = json.load(f)

    print(len(set([i['image_id'] for i in result_json])))
    print(len(set([i['image_id'] for i in validation_json['annotations']])))

    # assert set([i['image_id'] for i in result_json]) == set([i['image_id'] for i in validation_json['annotations']]),\
    #     "Images set mis-match: validation and result has different set of images"

    true_attributed_by_image_id, predicted_attributes_by_image_id = {}, {}
    for i in validation_json['annotations']:
        if i['image_id'] not in true_attributed_by_image_id:
            true_attributed_by_image_id.update({i['image_id']: [i['attribute_ids']]})
        else:
            true_attributed_by_image_id[i['image_id']].append(i['attribute_ids'])
    for result in result_json:
        if result['image_id'] not in predicted_attributes_by_image_id:
            predicted_attributes_by_image_id.update({result['image_id']: [result['attribute_ids']]})
        else:
            predicted_attributes_by_image_id[result['image_id']].append(result['attribute_ids'])
    del validation_json
    gc.collect()

    import pdb
    pdb.set_trace()

    y_true, y_pred = [], []
    for image_id, pred_attr in predicted_attributes_by_image_id.items():
        y_pred.append(pred_attr)
        y_true.append(true_attributed_by_image_id[image_id])

    import pdb
    pdb.set_trace()

    mlb = MultiLabelBinarizer()
    mlb.fit([np.arange(min(min(y_true)), max(max(y_true)))])

    score_f1 = f1_score(mlb.transform(y_true), mlb.transform(y_pred), average='macro')
    print(score_f1)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    app.run(main)
