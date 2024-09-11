import cv2
import numpy as np
from typing import Any
import time
import torch
import argparse
import yaml


model = None 


@torch.no_grad()
def benchmark(data : str, sample_size : int, model_name : str, jit_type : str):
    global model
    dts = []
        
    data = np.load("/root/catkin_ws/src/ssl_tests/data/" + data)
    
    #TODO: For data preprocessing 
    if model_name == "sam":
        pass
    if model_name == "dino":
        data = data[:, :224, :224, :]
        pass

    #TODO: For jit tracing/scripting
    if model_name == "sam":
        pass
    if model_name == "dino":
        if jit_type == "trace":
            image = torch.from_numpy(data[0]).permute(2,0,1).to(device="cuda")
            image = image.unsqueeze(0)
            image = image.type(torch.cuda.FloatTensor)
            model = torch.jit.trace(model, image)
        elif jit_type == "script":
            pass
        else:
            pass
            

    print("warming up..")
    for num in range(3):
        if model_name == "sam":
            image = torch.from_numpy(data[num]).permute(2,0,1).to(device="cuda")
            x = model.preprocess(image.unsqueeze(0))
            out = model.image_encoder(x)
            del x
            del out
        elif model_name == "dino":
            image = torch.from_numpy(data[num]).permute(2,0,1).to(device="cuda")
            image = image.unsqueeze(0)
            image = image.type(torch.cuda.FloatTensor)
            out = model(image)
            del out

    torch.cuda.synchronize()

    print("start timing..")
    for num in range(int(sample_size)):
        if model_name == "sam":
            image = torch.from_numpy(data[num]).permute(2,0,1).to(device="cuda")
            now = time.time()
            x = model.preprocess(image.unsqueeze(0))
            out = model.image_encoder(x)
            torch.cuda.synchronize()
            dts.append(time.time() - now)
            del x
        elif model_name == "dino":
            image = torch.from_numpy(data[num]).permute(2,0,1).to(device="cuda")
            image = image.unsqueeze(0)
            image = image.type(torch.cuda.FloatTensor)
            now = time.time()
            out = model(image)
            torch.cuda.synchronize()
            dts.append(time.time() - now)

        
        del out
        del image

    max_dt = max(dts)
    min_dt = min(dts)
    avg_dt = sum(dts)/len(dts)
    var_dt = sum((dt - avg_dt)**2 for dt in dts)/len(dts)

    if args.jit == "N/A": 
        new_data = {model_name :   
                              {"Time data" : 
                                          { "minimum": min_dt,
                                            "maximum": max_dt,
                                            "mean": avg_dt,
                                            "variance": var_dt,
                              }
                              }
                    }

    else:
        new_data = {model_name + args.jit:   
                              {"Time data" : 
                                          { "minimum": min_dt,
                                            "maximum": max_dt,
                                            "mean": avg_dt,
                                            "variance": var_dt,
                              }
                              }
                    }

    print(new_data)
    #Try reading and updating it
    try:
        with open("/root/catkin_ws/src/ssl_tests/data/inference_stats.yaml", 'r') as yaml_file:
            try:
                old_data = yaml.safe_load(yaml_file) or {}
                old_data.update(new_data)
            except yaml.YAMLError as exc:
                print(f"Error loading YAML file: {exc}")
                old_data = new_data 
    except FileNotFoundError:
        old_data = new_data

    #Try creating 
    try:
        with open("/root/catkin_ws/src/ssl_tests/data/inference_stats.yaml", 'w') as yaml_file:
            yaml.dump(old_data, yaml_file, default_flow_style=False)
    except yaml.YAMLError as exc:
        print(f"Error writing YAML file: {exc}")

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

    parser.add_argument(
        "--jit",
        type=str,
        default="N/A",
    )
    args = parser.parse_args()

    if args.model == "sam":
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        model = sam_model_registry["vit_h"](checkpoint="models/sam_vit_h_4b8939.pth").to(device="cuda")
        mask_generator = SamAutomaticMaskGenerator(model)
        
    elif args.model == "dino":

        # TODO: It seems that downloading backbones may have some control blocks that prevent tracing.
        # While loading directly via torch hub doesn't seem to have an issue.

        # from transformers import Dinov2Config, Dinov2Model
        # configuration = Dinov2Config()
        # model = Dinov2Model(configuration).to(device="cuda")
        # model.load_state_dict(torch.load("models/dinov2_vitb14_pretrain.pth"), strict = False)
        
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device = 'cuda')
        
    model.eval()
    benchmark(args.npy_file, args.sample_size, args.model, args.jit)