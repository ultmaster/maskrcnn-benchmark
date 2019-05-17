import os
import cv2 as cv

from maskrcnn_benchmark.config import cfg
from predictor import OpenImagesDemo

config_file = "../configs/test_openimages_single.yaml"

# update the config options with the config file
cfg.merge_from_file(config_file)
# manual override some options
# cfg.merge_from_list(["MODEL.DEVICE", "cpu"])
cfg.merge_from_list(["MODEL.WEIGHT", "../checkpoints/20190517_170829/model_0020000.pth"])

coco_demo = OpenImagesDemo(
    cfg,
    min_image_size=800,
    confidence_threshold=0.2,
)

for file_name in os.listdir("images"):
    # load image and then run prediction
    image = cv.imread(os.path.join("images", file_name))
    print("Running on", file_name)
    predictions = coco_demo.run_on_opencv_image(image)
    cv.imwrite(os.path.join("predictions", file_name), predictions)
