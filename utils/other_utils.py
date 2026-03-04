
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import os
import numpy as np
import json
import yaml
from pathlib import Path
        
def read_json_to_dic(file: str) -> dict:
    """ Create dictionnary from a json path

    Args:
        file (str): path to read from

    Returns:
        dict: output dic with json lib
    """
    assert(file.endswith(".json"))
    with open(file) as json_file:
        data = json.load(json_file)
    return data

def read_yaml_to_dic(file: str) -> dict:
    """ Create dictionnary from a yaml path

    Args:
        file (str): path to read from

    Returns:
        dict: output dic with yaml lib
    """
    assert(file.endswith(".yaml"))
    with open(file) as yaml_file:
        data = yaml.safe_load(Path(file).read_text())
    return data


def write_dic_to_yaml_file(dic, file):
    assert(file.endswith(".yaml"))
    with open(file, "w") as outfile:
        yaml.dump(dic, outfile, sort_keys=False, default_flow_style=False)