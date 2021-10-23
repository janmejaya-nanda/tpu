import json


with open('/home/user/impact/experiments_repo/fashionpedia/instances_attributes_val2020.json') as f:
    instances_attributes_val = json.load(f)
with open('/home/user/impact/experiments_repo/fashionpedia/instances_attributes_train2020.json') as f:
    instances_attributes_train = json.load(f)
with open('/home/user/impact/experiments_repo/fashionpedia/COCO_with_imageid2.json') as f:
    coco_result = json.load(f)

category_attributes_mapping = {}
for annotation in instances_attributes_train['annotations'] + instances_attributes_val['annotations']:
    if annotation['category_id'] not in category_attributes_mapping:
        category_attributes_mapping.update({annotation['category_id']: set(annotation['attribute_ids'])})
    else:
        category_attributes_mapping[annotation['category_id']] |= set(annotation['attribute_ids'])

for i in category_attributes_mapping:
    print(i)
    print("NUmber of attributes", len(category_attributes_mapping[i]))

with open('/home/user/impact/experiments_repo/fashionpedia/possible-category-attribute-mapping.json', 'w') as f:
    json.dump(category_attributes_mapping, f)

for result in coco_result:
    cat_id = result['category_id']
    result['attribute_ids'] = list(category_attributes_mapping[cat_id] & set(result['attribute_ids']))

with open('/home/user/impact/experiments_repo/fashionpedia/COCO_with_imageid2_filtered_for_non_mapped_attribute.json', 'w') as f:
    json.dump(coco_result, f)
