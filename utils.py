import os
import glob
import numpy as np
import torch


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
    ultrasound_filenames = glob.glob(os.path.join(data_arrays_fullpath, "*ultrasound*.npy"))
    segmentation_filenames = glob.glob(os.path.join(data_arrays_fullpath, "*segmentation*.npy"))
    subject_ids = set([int(os.path.splitext(os.path.basename(fn))[0].split("-")[1]) for fn in ultrasound_filenames])
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
        subject_ultrasound_indices = [idx for idx, filename in enumerate(ultrasound_filenames) 
                                        if int(os.path.splitext(os.path.basename(filename))[0].split("-")[1]) == subject_id]
        for idx in subject_ultrasound_indices:
            subject_ultrasound_array = np.concatenate([subject_ultrasound_array, ultrasound_arrays[idx]])
        
        subject_segmentation_indices = [idx for idx, filename in enumerate(segmentation_filenames)
                                        if int(os.path.splitext(os.path.basename(filename))[0].split("-")[1]) == subject_id]
        for idx in subject_segmentation_indices:
            subject_segmentation_array = np.concatenate([subject_segmentation_array, segmentation_arrays[idx]])

        ultrasound_arrays_by_subjects.append(subject_ultrasound_array)
        segmentation_arrays_by_subjects.append(subject_segmentation_array)

    return ultrasound_arrays_by_subjects, segmentation_arrays_by_subjects


def get_data_array(data_array_by_subject, patient_idx):
    data_array = np.zeros([
        0,
        data_array_by_subject[0].shape[1],
        data_array_by_subject[0].shape[2],
        data_array_by_subject[0].shape[3]]
    )
    for idx in patient_idx:
        data_array = np.concatenate([data_array, data_array_by_subject[idx]])
    return data_array


def list_images_by_subject(ultrasound_arrays_by_subjects, segmentation_arrays_by_subjects):
    total_ultrasound_count = 0
    total_segmentation_count = 0
    for i in range(len(ultrasound_arrays_by_subjects)):
        total_ultrasound_count += ultrasound_arrays_by_subjects[i].shape[0]
        print("Subject {}: {} images and {} segmentations".format(
            i, ultrasound_arrays_by_subjects[i].shape[0], segmentation_arrays_by_subjects[i].shape[0]))
    print("Total images: {}".format(total_ultrasound_count))
    print("Total segmentations: {}".format(total_segmentation_count))


if __name__ == "__main__":
    project_path = "d:/UltrasoundSegmentation"
    us_arrays, seg_arrays = load_ultrasound_data(os.path.join(project_path, "DataArrays"))
    list_images_by_subject(us_arrays, seg_arrays)
