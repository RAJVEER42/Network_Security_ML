import yaml
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import os,sys
import numpy as np
#import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def read_yaml_file(file_path: str) -> dict:  # return in Dir form buz the data is in key & values pairs
    try:
        with open(file_path, "rb") as yaml_file:    # reading the yaml file
            return yaml.safe_load(yaml_file)        # loading the yaml file
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path): # if exist then remove the file
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    