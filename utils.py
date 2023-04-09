import os
import glob
import numpy as np
from more_itertools import set_partitions


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
        subject_indices = [idx for idx, filename in enumerate(ultrasound_filenames) if str(subject_id) in filename]
        for i in range(len(subject_indices)):
            subject_ultrasound_array = np.concatenate([subject_ultrasound_array, ultrasound_arrays[i]])
            subject_segmentation_array = np.concatenate([subject_segmentation_array, segmentation_arrays[i]])

        ultrasound_arrays_by_subjects.append(subject_ultrasound_array)
        segmentation_arrays_by_subjects.append(subject_segmentation_array)

    return ultrasound_arrays_by_subjects, segmentation_arrays_by_subjects


def get_train_test_val_indices(arrays_by_patient, train_split=0.8, val_split=0.1, test_split=0.1):
    assert train_split + val_split + test_split == 1.0

    # Get ideal number of images in each partition
    arr_shapes = [arr.shape[0] for arr in arrays_by_patient]
    total_img_count = sum(arr_shapes)
    img_split_counts = (total_img_count * train_split, total_img_count * val_split, total_img_count * test_split)

    # Get the closest partition of the images by patient
    all_partitions = list(set_partitions(arr_shapes, 3))
    best_partition = all_partitions[0]
    i = 1
    while i < len(all_partitions):
        if abs(sum(all_partitions[i][0]) - img_split_counts[0]) < abs(sum(best_partition[0]) - img_split_counts[0]) \
            and abs(sum(all_partitions[i][1]) - img_split_counts[1]) < abs(sum(best_partition[1]) - img_split_counts[1]) \
            and abs(sum(all_partitions[i][2]) - img_split_counts[2]) < abs(sum(best_partition[2]) - img_split_counts[2]):
            best_partition = all_partitions[i]
        i += 1
    
    # Find indices of closest partition
    train_indices = [arr_shapes.index(i) for i in best_partition[0]]
    val_indices = [arr_shapes.index(i) for i in best_partition[1]]
    test_indices = [arr_shapes.index(i) for i in best_partition[2]]
    assert train_indices + val_indices + test_indices == len(arr_shapes)

    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    project_path = "e:/PerkLab/Data/BreastSurgery/BreastUltrasound"
    us_arrays, seg_arrays = load_ultrasound_data(os.path.join(project_path, "DataArrays"))
    train_indices, val_indices, test_indices = get_train_test_val_indices(us_arrays)
    print(train_indices, val_indices, test_indices)
    train_data = us_arrays[train_indices]
    val_data = us_arrays[val_indices]
    test_data = us_arrays[test_indices]
    print(sum([arr.shape[0] for arr in train_data]))
    print(sum([arr.shape[0] for arr in val_data]))
    print(sum([arr.shape[0] for arr in test_data]))
