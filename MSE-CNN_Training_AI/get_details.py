import tensorflow as tf
import net_cu as net
import h5py
import input_data_H5 as input_data
import warnings
warnings.filterwarnings("ignore")

CU_WIDTH_LIST = [64,  32, 16, 32, 8, 32, 16, 8, 32, 16]
CU_HEIGHT_LIST = [64, 32, 16, 16, 8, 8,  8,  4, 4,   4]

LABEL_LENGTH_LIST = [2, 6, 6, 6, 6, 6, 6, 6, 6, 6]
SAMPLE_LENGTH_LIST = [4105, 1033, 265, 521, 73, 265, 137, 41, 137, 73]
IMAGES_LENGTH_LIST = [4096, 1024, 256, 512, 64, 256, 128, 32, 128, 64]

TRAIN_SAMPLE_AMOUNT_LIST = [600,  0,    0,   0,   0,  0,   0,   0,  0,   0 ]
VALID_SAMPLE_AMOUNT_LIST = [600,  0,    0,   0,   0,  0,   0,   0,  0,   0 ]

def get_sample_details(index):
    size_train = TRAIN_SAMPLE_AMOUNT_LIST[index]
    size_valid = VALID_SAMPLE_AMOUNT_LIST[index]
    CU_WIDTH = CU_WIDTH_LIST[index]
    CU_HEIGHT = CU_HEIGHT_LIST[index]
    IMAGES_LENGTH = IMAGES_LENGTH_LIST[index]
    # SAMPLE_LENGTH = SAMPLE_LENGTH_LIST[index]
    LABEL_LENGTH = LABEL_LENGTH_LIST[index]

    return size_train, size_valid, CU_WIDTH, CU_HEIGHT, IMAGES_LENGTH, LABEL_LENGTH