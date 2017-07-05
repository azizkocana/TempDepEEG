import numpy as np


def convert_data(X_py, folder):
    """ Saves provided data as npy format
    Args:
        X_py(ndarray): data in numpy format
        folder(str): folder name
    """

    dat = np.asarray(X_py)
    np.save(folder, dat)
