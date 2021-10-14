import json


with open('/home/user/impact/experiments_repo/fashionpedia/COCO_with_imageid2.json') as f:
    result_json = json.load(f)
with open('/home/user/impact/experiments_repo/fashionpedia/instances_attributes_val2020.json') as f:
    validation_json = json.load(f)

predicted_attribute_count = {}
for i in result_json:
    for attribute_id in i['attribute_ids']:
        if attribute_id in predicted_attribute_count:
            predicted_attribute_count[attribute_id] += 1
        else:
            predicted_attribute_count.update({attribute_id: 1})
true_attribute_count = {}
for i in validation_json['annotations']:
    for attribute_id in i['attribute_ids']:
        if attribute_id in true_attribute_count:
            true_attribute_count[attribute_id] += 1
        else:
            true_attribute_count.update({attribute_id: 1})

# Scores
for i in predicted_attribute_count:
    acc_score = 1 - (abs(predicted_attribute_count[i] - true_attribute_count[i]) / true_attribute_count[i]) if i in \
                                                                                                            true_attribute_count else 0
    print("Attribute ID: {}, Score: {}".format(i, acc_score))

import pdb
pdb.set_trace()