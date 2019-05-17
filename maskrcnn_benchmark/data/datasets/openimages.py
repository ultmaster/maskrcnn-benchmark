import csv
import os
import sys
from collections import defaultdict

import torch
from PIL import Image
from torch.utils import data

from maskrcnn_benchmark.structures.bounding_box import BoxList

class OpenImagesDataset(data.Dataset):
    def __init__(self, ann_file, class_descriptions_file, valid_image_list_file,
                 root, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.root = root

        available_images = dict()
        class_labels = dict()
        available_bbox = defaultdict(list)

        print("Loading Open Images Dataset...")
        sys.stdout.flush()
        with open(valid_image_list_file) as f:
            csv_f = csv.reader(f)  # no header row
            for i, row in enumerate(csv_f):
                if i % 100000 == 0:
                    print("Valid images list file progress:", i)
                    sys.stdout.flush()
                available_images[row[0]] = (int(row[1]), int(row[2]))

        with open(class_descriptions_file) as f:
            for i, row in enumerate(csv.reader(f)):
                class_labels[row[0]] = i + 1

        with open(ann_file) as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0: continue  # skip header row
                if i % 100000 == 0:
                    print("Annotation list file load progress:", i)
                    sys.stdout.flush()
                key = row[0]
                if key not in available_images:
                    continue
                label = class_labels[row[2]]
                width, height = available_images[key]
                available_bbox[key].append([float(row[4]) * width, float(row[6]) * width,
                                            float(row[5]) * height, float(row[7]) * height,
                                            label])

        self.image_keys = sorted(available_bbox.keys())
        self.image_bbox = [[lst[:4] for lst in available_bbox[key][:4]] for key in self.image_keys]
        self.image_labels = [[lst[4] for lst in available_bbox[key][:4]] for key in self.image_keys]
        self.image_sizes = [available_images[key] for key in self.image_keys]
        print("Index created. Dataset size = %d" % len(self.image_keys))
        sys.stdout.flush()

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        key = self.image_keys[idx]
        # load the image as a PIL Image
        image = Image.open(os.path.join(self.root, key + ".jpg")).convert('RGB')

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use
        # x1, y1, x2, y2 order.
        boxes = self.image_bbox[idx]
        # and labels
        labels = torch.tensor(self.image_labels[idx])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.size, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.image_sizes[idx][1],
                "width": self.image_sizes[idx][0]}
