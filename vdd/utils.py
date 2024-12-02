from typing import Dict
# from GPUtil import GPUtil
import os
import copy
import shutil
import yaml

from pathlib import Path
from datetime import datetime

import logging

import pathlib

import torch
import numpy as np


def global_seeding(seed: int):
    """
    Set the seed for numpy and torch
    Args:
        seed: seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)  # Sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs.
    torch.backends.cudnn.deterministic = True  # Forces deterministic algorithm selections for convolution.
    torch.backends.cudnn.benchmark = False  # Disables the inbuilt cudnn auto-tuner.
    print(f"Setting global seed: {seed}")


# def get_free_gpus():
#     return GPUtil.getAvailable(order='memory', limit=1, maxLoad=0.8, maxMemory=0.8, includeNan=False,
#                                     excludeID=[], excludeUUID=[])

def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)

def mkdir(directory: str, overwrite: bool = False):
    """

    Args:
        directory: dir path to make
        overwrite: overwrite exist dir

    Returns:
        None

    Raise:
        FileExistsError if dir exists and overwrite is False
    """
    path = Path(directory)
    try:
        path.mkdir(parents=True, exist_ok=overwrite)
    except FileExistsError:
        logging.error("Directory already exists, remove it before make a new one.")
        raise

def set_value_in_nest_dict(config, key, value):
    """
    Set value of a certain key in a recursive way in a nested dictionary

    Args:
        config: configuration dictionary
        key: key to ref
        value: value to set

    Returns:
        config
    """
    for k in config.keys():
        if k == key:
            config[k] = value
        if isinstance(config[k], dict):
            set_value_in_nest_dict(config[k], key, value)
    return config

def remove_file_dir(path: str) -> bool:
    """
    Remove file or directory
    Args:
        path: path to directory or file

    Returns:
        True if successfully remove file or directory

    """
    if not os.path.exists(path):
        return False
    elif os.path.isfile(path) or os.path.islink(path):
        os.unlink(path)
        return True
    else:
        shutil.rmtree(path)
        return True


def dump_config(config_dict: dict, config_name: str, dump_dir: str):
    """
    Dump configuration into yaml file
    Args:
        config_dict: config dictionary to be dumped
        config_name: config file name
        dump_dir: dir to dump
    Returns:
        None
    """

    # Generate config path
    dump_path = os.path.join(dump_dir, config_name + ".yaml")

    # Remove old config if exists
    # remove_file_dir(dump_path)

    # Write new config to file
    with open(dump_path, "w") as f:
        yaml.dump(config_dict, f)

def dir_go_up(num_level: int = 2, current_file_dir: str = "default") -> str:
    """
    Go to upper n level of current file directory
    Args:
        num_level: number of level to go up
        current_file_dir: current dir

    Returns:
        dir n level up
    """
    if current_file_dir == "default":
        current_file_dir = os.path.realpath(__file__)
    while num_level != 0:
        current_file_dir = os.path.dirname(current_file_dir)
        num_level -= 1
    return current_file_dir


def get_formatted_date_time() -> str:
    """
    Get formatted date and time, e.g. May-01-2021 22:14:31
    Returns:
        dt_string: date time string
    """
    now = datetime.now()
    dt_string = now.strftime("%b-%2d-%Y-%H:%M:%S")
    return dt_string

def make_log_dir_with_time_stamp(log_name: str) -> str:
    """
    Get the dir to the log
    Args:
        log_name: log's name

    Returns:
        directory to log file
    """

    return os.path.join(dir_go_up(3), "log", log_name,
                        get_formatted_date_time())

def process_cw2_train_rep_config_file(config_obj, overwrite: bool = False):
    """
    Given processed cw2 configuration, do further process, including:
    - Overwrite log path with time stamp
    - Create model save folders
    - Overwrite random seed by the repetition number
    - Save the current repository commits
    - Make a copy of the config and restore the exp path to the original
    - Dump this copied config into yaml file into the model save folder
    - Dump the current time stamped config file in log folder to make slurm
      call bug free
    Args:
        exp_configs: list of configs processed by cw2 already

    Returns:
        None

    """
    exp_configs = config_obj.exp_configs
    formatted_time = get_formatted_date_time()
    # Loop over the config of each repetition
    for i, rep_config in enumerate(exp_configs):

        # Make model save directory
        model_save_dir = os.path.join(rep_config["_rep_log_path"], "model")

        try:
            mkdir(os.path.abspath(model_save_dir), overwrite=overwrite)
        except FileExistsError:
            import logging
            logging.error(formatted_time)
            raise

        # Set random seed to the repetition number
        set_value_in_nest_dict(rep_config, "seed",
                                    rep_config['_rep_idx'])


        # Make a hard copy of the config
        copied_rep_config = copy.deepcopy(rep_config)

        # Recover the path to its original
        copied_rep_config["path"] = copied_rep_config["_basic_path"]

        # Reset the repetition number to 1 for future test usage
        copied_rep_config["repetitions"] = 1
        if copied_rep_config.get("reps_in_parallel", False):
            del copied_rep_config["reps_in_parallel"]
        if copied_rep_config.get("reps_per_job", False):
            del copied_rep_config["reps_per_job"]

        # Delete the generated cw2 configs
        for key in rep_config.keys():
            if key[0] == "_":
                del copied_rep_config[key]
        del copied_rep_config["log_path"]

        # Save this copied subconfig file
        dump_config(copied_rep_config, "config",
                         os.path.abspath(model_save_dir))

    # Save the time stamped config file in local /log directory
    time_stamped_config_path = make_log_dir_with_time_stamp("")
    mkdir(time_stamped_config_path, overwrite=True)

    config_obj.to_yaml(time_stamped_config_path,
                       relpath=False)
    config_obj.config_path = \
        os.path.join(time_stamped_config_path,
                       "relative_" + config_obj.f_name)

def load_metadata_from_yaml(path: str, file_name: str = 'config.yaml') -> Dict:
    """
    Load meta data from yaml file
    """
    load_path = pathlib.Path(path, file_name)
    with open(load_path, 'r') as f:
        try:
            meta_data = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            print(exc)
    return meta_data


def sort_filenames(filenames):
    # Function to extract the numerical part from the filename
    def extract_number(filename):
        # This assumes the format is always "model_state_dict_NUMBER.pth"
        k = int(filename.split('.')[0].split('_')[-1])
        return k

    sorted_filenames = sorted(filenames, key=extract_number)
    return sorted_filenames


def get_files_with_prefix(path, prefix):
    file_name_list = os.listdir(path)
    file_name_list = [file_name for file_name in file_name_list if file_name.startswith(prefix)]
    return sort_filenames(file_name_list)