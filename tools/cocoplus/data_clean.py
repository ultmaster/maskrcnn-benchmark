import json
import os

from PIL import Image
from pycocotools.coco import COCO

with open("datasets/cocoplus/annotations_raw.json", "r") as f:
    data = json.load(f)

coco = COCO("datasets/cocoplus/annotations_raw.json")
image_dir = "datasets/cocoplus/images"
images_data, anno_data = [], []
images_id_set = set()
for id in coco.getImgIds():
    img_dict = coco.loadImgs(id)[0]
    img_path = os.path.join(image_dir, img_dict["file_name"])
    try:
        img = Image.open(img_path).convert("RGB")
        images_data.append(img_dict)
        images_id_set.add(img_dict["id"])
    except:
        print(img_dict["file_name"], img_dict["id"])
for anno in coco.getAnnIds():
    anno_dict = coco.loadAnns(anno)[0]
    anno_dict["area"] = anno_dict["bbox"][2] * anno_dict["bbox"][3]
    if anno_dict["image_id"] in images_id_set:
        anno_data.append(anno_dict)
data["images"] = images_data
data["annotations"] = anno_data
with open("datasets/cocoplus/annotations.json", "w") as f:
    json.dump(data, f)
