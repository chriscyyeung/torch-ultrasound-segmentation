import os
import glob
import numpy as np


def create_standard_project_folders(local_data_folder):
    # These subfolders will be created/populated in the data folder
    data_arrays_folder    = "DataArrays"
    results_save_folder   = "SavedResults"
    models_save_folder    = "SavedModels"
    logs_save_folder      = "Logs"

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


def load_ultrasound_data(data_arrays_fullpath):
    ultrasound_filenames = glob.glob(os.path.join(data_arrays_fullpath, "ultrasound*.npy"))
    segmentation_filenames = glob.glob(os.path.join(data_arrays_fullpath, "segmentation*.npy"))
    subject_ids = set([int(os.path.splitext(os.path.basename(fn)).split("-")[2]) for fn in ultrasound_filenames])
    n_arrays = len(subject_ids)

    # Load arrays from local files
    ultrasound_arrays = []
    segmentation_arrays = []
    for i in range(n_arrays):
        ultrasound_data = np.load(ultrasound_filenames[i])
        segmentation_data = np.load(segmentation_filenames[i])

        ultrasound_arrays.append(ultrasound_data)
        segmentation_arrays.append(segmentation_data)

    # Concatenate arrays by subjects (e.g. patients)
    ultrasound_arrays_by_subjects = []
    segmentation_arrays_by_subjects = []

    ultrasound_pixel_type = ultrasound_arrays[0].dtype
    segmentation_pixel_type = segmentation_arrays[0].dtype

    for subject_id in subject_ids:
        subject_ultrasound_array = np.zeros([0, ultrasound_arrays[0].shape[1], ultrasound_arrays[0].shape[2], 1],
                                            dtype=ultrasound_pixel_type)

        subject_segmentation_array = np.zeros([0, segmentation_arrays[0].shape[1], segmentation_arrays[0].shape[2], 1],
                                              dtype=segmentation_pixel_type)

        # Combine arrays from the same subject
        subject_indices = [idx for idx, filename in enumerate(ultrasound_filenames) if str(subject_id) in filename]
        for i in range(len(subject_indices)):
            subject_ultrasound_array = np.concatenate([subject_ultrasound_array, ultrasound_arrays[i]])
            subject_segmentation_array = np.concatenate([subject_segmentation_array, segmentation_arrays[i]])

        ultrasound_arrays_by_subjects.append(subject_ultrasound_array)
        segmentation_arrays_by_subjects.append(subject_segmentation_array)

    return ultrasound_arrays_by_subjects, segmentation_arrays_by_subjects
