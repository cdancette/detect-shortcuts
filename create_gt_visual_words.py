import json
from collections import defaultdict
from tqdm import tqdm


with open("/data/common-data/coco/annotations/instances_train2014.json") as f:
    data_train = json.load(f)
with open("/data/common-data/coco/annotations/instances_val2014.json") as f:
    data_val = json.load(f)

result_path = "data/image_to_gt_vocab.json"

result = defaultdict(lambda: {"classes": [], "scores": []})

categories = {c["id"]: c for c in data_train["categories"]}

for annot in tqdm(data_train["annotations"]):
    image_id = annot["image_id"]
    category = categories[annot["category_id"]]["name"]
    if category not in result[image_id]["classes"]:
        result[image_id]["classes"].append(category)
        result[image_id]["scores"].append(1.0)
print(len(result))

for annot in tqdm(data_val["annotations"]):
    image_id = annot["image_id"]
    category = categories[annot["category_id"]]["name"]
    if category not in result[image_id]["classes"]:
        result[image_id]["classes"].append(category)
        result[image_id]["scores"].append(1.0)
print(len(result))

with open(result_path, "w") as f:
    json.dump(result, f)
