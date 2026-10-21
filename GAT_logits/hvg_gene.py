def extract_hvg_and_split(file_path):
    import h5py
    import numpy as np
    import pandas as pd

    with h5py.File(file_path, 'r') as f:
        X_train_full = np.array(f['train_matrix'])
        X_val_full = np.array(f['val_matrix'])
        X_test_full = np.array(f['test_matrix'])

        X_train_mask = np.array(f['train_mask_matrix'])
        X_val_mask = np.array(f['val_mask_matrix'])
        X_test_mask = np.array(f['test_mask_matrix'])

    X_train = X_train_full * X_train_mask
    X_val = X_val_full * X_val_mask
    X_test = X_test_full * X_test_mask

    spatial_train = pd.read_csv('D:/stDGA/data/Sporadic_data/decidua basalis_10%/train_location.csv')
    spatial_val = pd.read_csv('D:/stDGA/data/Sporadic_data/decidua basalis_10%/val_location.csv')
    spatial_test = pd.read_csv('D:/stDGA/data/Sporadic_data/decidua basalis_10%/test_location.csv')

    labels_train = pd.read_csv("D:/stDGA/data/Sporadic_data/decidua basalis_10%/train_labels.csv")
    labels_val = pd.read_csv("D:/stDGA/data/Sporadic_data/decidua basalis_10%/val_labels.csv")
    labels_test = pd.read_csv("D:/stDGA/data/Sporadic_data/decidua basalis_10%/test_labels.csv")

    return {
        "train": (X_train, X_train_full, X_train_mask, spatial_train, labels_train),
        "val": (X_val, X_val_full, X_val_mask, spatial_val, labels_val),
        "test": (X_test, X_test_full, X_test_mask, spatial_test, labels_test)
    }
