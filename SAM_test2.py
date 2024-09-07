# import rospy, make it subscribe to "camera/color/image_raw" topic and run sam on the image
# overlay the mask on the image and publish it to "camera/color/image_masked" topic

import rospy
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import time
import torch
import argparse
import yaml

sam = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").to(device="cuda")
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)
bridge = CvBridge()


@torch.no_grad()
def test(data : str, sample_size : int):
    global sam
    print("got image")
    data = np.load("/root/catkin_ws/src/ssl_tests/data/" + data)
    for dt in range(int(sample_size)):
        now = time.time()
        image = torch.from_numpy(data[dt]).permute(2,0,1).to(device="cuda")
        x = sam.preprocess(image.unsqueeze(0))
        out = sam.image_encoder(x)
        print(time.time() - now)
        del x
        del out
        del image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npy_file",
        type=str,
        default="hound_25.npy",
    )
    parser.add_argument(
        "--sample_size",
        type=str,
        default="10",
    )
    args = parser.parse_args()
    test(args.npy_file, args.sample_size)