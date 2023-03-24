import os
import numpy as np
import pandas as pd


def create_standard_project_folders(local_data_folder):
    # These subfolders will be created/populated in the data folder
    data_arrays_folder    = "DataArrays"
    notebooks_save_folder = "SavedNotebooks"
    results_save_folder   = "SavedResults"
    models_save_folder    = "SavedModels"
    logs_save_folder      = "Logs"
    val_data_folder       = "PredictionsValidation"

    data_arrays_fullpath = os.path.join(local_data_folder, data_arrays_folder)
    results_save_fullpath = os.path.join(local_data_folder, results_save_folder)
    models_save_fullpath = os.path.join(local_data_folder, models_save_folder)
    logs_save_fullpath = os.path.join(local_data_folder, logs_save_folder)

    if not os.path.exists(data_arrays_fullpath):
        os.makedirs(data_arrays_fullpath)
        print("Created folder: {}".format(data_arrays_fullpath))

    if not os.path.exists(results_save_fullpath):
        os.makedirs(results_save_fullpath)
        print("Created folder: {}".format(results_save_fullpath))

    if not os.path.exists(models_save_fullpath):
        os.makedirs(models_save_fullpath)
        print("Created folder: {}".format(models_save_fullpath))

    if not os.path.exists(logs_save_fullpath):
        os.makedirs(logs_save_fullpath)
        print("Created folder: {}".format(logs_save_fullpath))
    
    return data_arrays_fullpath, results_save_fullpath, models_save_fullpath, logs_save_fullpath
