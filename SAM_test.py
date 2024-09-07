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

rospy.init_node('sam_test')
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device="cuda")
sam.eval()
mask_generator = SamAutomaticMaskGenerator(sam)
bridge = CvBridge()

@torch.no_grad()
def image_callback(data):
    global sam
    print("got image")
    now = time.time()
    image = bridge.imgmsg_to_cv2(data, "bgr8")
    image = torch.from_numpy(image).permute(2,0,1).to(device="cuda")
    x = sam.preprocess(image.unsqueeze(0))
    out = sam.image_encoder(x)
    print(time.time() - now)
    del x
    del out
    del image
    # masked_image = cv2.bitwise_and(image, image, mask=masks[0])
    # masked_image = bridge.cv2_to_imgmsg(masked_image, "bgr8")
    # cv2.imshow("masked_image", masked_image)
    # cv2.waitKey(1)

rospy.Subscriber("/camera/color/image_raw", Image, image_callback)

while not rospy.is_shutdown():
    rospy.spin()
