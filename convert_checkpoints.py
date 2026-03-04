import torch
from utils.other_utils import read_yaml_to_dic, write_dic_to_yaml_file
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", "-ip", type=str, required=True)
    args = parser.parse_args()

    input_path = args.input_path
    assert input_path.endswith(".pt") or input_path.endswith(".pth"), "Input path must end with .pt or .pth"
    assert os.path.exists(input_path), "Input path does not exist"
    input_dic = torch.load(input_path, weights_only=False)


    model_state_dict = input_dic['model_state_dict']
    config = None
    if "config" in input_dic:
        config = input_dic["config"]
    elif "hyperparameters" in input_dic:
        config = input_dic["hyperparameters"]
    else:
        raise ValueError("Config not found in input dictionary")

    val_auc = input_dic.get("val_auc", 0.0)
    val_ap = input_dic.get("val_ap", 0.0)
    
    meta = {"config": config, "val_auc": val_auc, "val_ap": val_ap}

    # Convert to separate files
    output_name = os.path.basename(input_path).split(".")[0]
    output_path = os.path.join(os.path.dirname(input_path), "converted_"+output_name)
    os.makedirs(output_path, exist_ok=True)
    
    # in the dir save separately the model_state_dict and the config
    torch.save(model_state_dict, os.path.join(output_path, "model_state_dict.pt"))
    write_dic_to_yaml_file(meta, os.path.join(output_path, "meta.yaml"))
    print(f"Checkpoint converted and saved to {output_path}")


