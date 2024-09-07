#!/usr/bin/env python3
import rosbag
import os
import argparse
import numpy as np
import cv2
from cv_bridge import CvBridge
bridge = CvBridge()

def main(file_path : str, sample_size : int, topic_name : str):

    # Load the bag file
    bag = rosbag.Bag("/root/catkin_ws/src/bags/" + file_path, "r")

    # Get the first message
    images = []
    cnt = 0
    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if topic == "/camera/color/image_raw":
            if (cnt == sample_size):
                break
            images.append(bridge.imgmsg_to_cv2(msg, "bgr8"))
            cnt+=1
    print("Writing..")
    np.save("/root/catkin_ws/src/ssl_tests/data/" +  file_path.rstrip(".bag") + ".npy", images)


if __name__ == "__main__":
    print(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag_file",
        type=str,
        default="hound_25.bag",
    )
    parser.add_argument(
        "--sample_size",
        type=str,
        default="10",
    )
    args = parser.parse_args()

    main(args.bag_file, args.sample_size, "/camera/color/image_raw")
