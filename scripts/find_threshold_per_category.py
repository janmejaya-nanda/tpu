import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


IOU_THRESHOLD = 0.5
N_ATTRIBUTE_THRESHOLD = 50
NUMBER_OF_CATEGORY = 46


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


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    c = 0
    if bb2['y1'] > bb2['y2']:
        c, bb2['y2'] = bb2['y2'], bb2['y1']
        bb2['y1'] = c
    if bb1['y1'] > bb1['y2']:
        c, bb1['y2'] = bb1['y2'], bb1['y1']
        bb1['y1'] = c
    if bb2['x1'] > bb2['x2']:
        c, bb2['x2'] = bb2['x2'], bb2['x1']
        bb2['x1'] = c
    if bb1['x1'] > bb1['x2']:
        c, bb1['x2'] = bb1['x2'], bb1['x1']
        bb1['x1'] = c

    assert bb1['x1'] <= bb1['x2']
    assert bb1['y1'] <= bb1['y2']
    assert bb2['x1'] <= bb2['x2']
    assert bb2['y1'] <= bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# with open('/home/user/impact/experiments_repo/fashionpedia/instances_attributes_val2020.json') as f:
#     instances_attributes = json.load(f)
# with open('/home/user/impact/experiments_repo/fashionpedia/label_descriptions.json') as f:
#     data = json.load(f)
#     attributes_map = dict([(attribute['id'], attribute['name']) for attribute in data['attributes']])
#     attribute_ids = np.array([i[0] for i in sorted(attributes_map.items(), key=lambda x: x[0])])
#
# data = np.load('/home/user/impact/experiments_repo/fashionpedia/output.npy', allow_pickle=True)
#
# annotations_df = pd.DataFrame(instances_attributes['annotations'])
# images_df = pd.DataFrame(instances_attributes['images'])
# images_annotations = images_df.merge(annotations_df, how="inner", left_on='id', right_on='image_id',
#                                      suffixes=(None, "_y"), sort=True)
# images_annotations_grp = images_annotations.groupby(['image_id'], as_index=False)
# file_name_by_image_id = dict(zip(images_annotations['file_name'].tolist(), images_annotations['image_id'].tolist()))
# data = [{**i, 'image_id': file_name_by_image_id[i['image_file'].split('/')[-1]]} for i in data if i[
#     'image_file'].split('/')[-1] in file_name_by_image_id]
#
# min_attribute_score, max_attribute_score = 1, 0
# for i in data:
#     if i['attributes_prob'].min() < min_attribute_score:
#         min_attribute_score = i['attributes_prob'].min()
#     if i['attributes_prob'].max() > max_attribute_score:
#         max_attribute_score = i['attributes_prob'].max()
# increment = (max_attribute_score - min_attribute_score) / N_ATTRIBUTE_THRESHOLD
# attribute_score_range = np.arange(min_attribute_score, max_attribute_score, increment)
#
# f1_by_threshold = dict([(attribute_threshold, np.zeros(NUMBER_OF_CATEGORY)) for attribute_threshold in attribute_score_range])
# for attribute_threshold in attribute_score_range:
#     for category_id in range(NUMBER_OF_CATEGORY):
#         y_true, y_pred = [], []
#         for result in data:
#             for box, result_category_id, attribute_score, score in zip(result['boxes'], result['classes'],
#                                                                  result['attributes_prob'], result['scores']):
#                 if (result_category_id - 1) != category_id:
#                     continue
#                 ind = np.argwhere(attribute_score > attribute_threshold).ravel()
#                 predicted_attribute_ids = list(attribute_ids[ind])
#
#                 # find suitable annotation by finding overlapping IOU and category ID
#                 annotations = images_annotations_grp.get_group(result['image_id'])
#                 for box2, annotation_category_id, annotation_attribute_ids in zip(annotations['bbox'].tolist(), annotations[
#                     'category_id'].tolist(), annotations['attribute_ids'].tolist()):
#                     iou = get_iou(bb1={'x1': box[1], 'y1': box[0], 'x2': box[3], 'y2': box[2]},
#                                   bb2={'x1': box2[1], 'y1': box2[0], 'x2': box2[3], 'y2': box2[2]})
#
#                     # modifying result category to 0th id
#                     if iou > IOU_THRESHOLD and ((result_category_id - 1) == annotation_category_id):
#                         y_true.append(annotation_attribute_ids)
#                         y_pred.append(predicted_attribute_ids)
#                 else:
#                     # Unable to find suitable annotation
#                     y_true.append([])
#                     y_pred.append(predicted_attribute_ids)
#
#         mlb = MultiLabelBinarizer()
#         mlb.fit([np.arange(0, 340)])
#
#         score_f1 = f1_score(mlb.transform(y_true), mlb.transform(y_pred), average='macro')
#         print(score_f1)
#
#         f1_by_threshold[attribute_threshold][category_id] = score_f1
#
# with open('/home/user/impact/experiments_repo/fashionpedia/f1_by_threshold.json', 'w') as f:
#     json.dump(f1_by_threshold, f, cls=NpEncoder)

with open('/home/user/impact/experiments_repo/fashionpedia/f1_by_threshold.json') as f:
    f1_by_threshold = json.load(f)

thresholds, scores = np.array([]), np.array([])
for t, score in f1_by_threshold.items():
    thresholds = np.append(thresholds, [t])
    if len(scores):
        scores = np.vstack((scores, score))
    else:
        scores = score

print("Heighest F1 Score:", scores.max(axis=0))
print("Thresholds:", [round(i, 6) for i in map(float, thresholds[scores.argmax(axis=0)])])
