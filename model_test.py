# import rospy, make it subscribe to "camera/color/image_raw" topic and run sam on the image
# overlay the mask on the image and publish it to "camera/color/image_masked" topic

import rospy
import cv2
import numpy as np
from typing import Any
import time
import torch
import argparse
import yaml


model = None


@torch.no_grad()
def test(model : Any, data : str, sample_size : int, model_name : str):
    global sam
    dts = []
    data = np.load("/root/catkin_ws/src/ssl_tests/data/" + data)
    for num in range(int(sample_size)):
        now = time.time()
        image = torch.from_numpy(data[num]).permute(2,0,1).to(device="cuda")
        x = sam.preprocess(image.unsqueeze(0))
        out = sam.image_encoder(x)
        dts.append(time.time() - now)
        del x
        del out
        del image

    max_dt = max(dts)
    min_dt = min(dts)
    avg_dt = sum(dts)/len(dts)
    tru_avg_dt = avg_dt - max_dt/(len(dts))
    var_dt = sum((dt - tru_avg_dt)**2 for dt in dts)/len(dts)

    new_data  = {
            "model" : model_name,
                "inference time data" : {
                    "minimum" : min_dt,
                    "maximum" : max_dt,
                    "mean"    : tru_avg_dt,
                    "variance": var_dt,
                }
            }
    with open("/root/catkin_ws/src/ssl_tests/data/stats.yaml", 'r') as  yaml_file:
        try:
            old_data = yaml.safe_load(yaml_file)
            old_data.update(new_data)
            yaml.dump(new_data, yaml_file, default_flow_style=False)
        except yaml.YAMLError as exc:
            print(f"Error loading YAML file: {exc}")
            print(new_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="sam",
    )
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

    if args.model == "sam":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        model = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").to(device="cuda")
        model.eval()
        mask_generator = SamAutomaticMaskGenerator(model)

    elif args.model == "dino":
        pass


    test(model, args.npy_file, args.sample_size, args.model)