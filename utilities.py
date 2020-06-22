import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

from keras.applications.nasnet import NASNetLarge

# Read settings CSV
def read_settings(path):
    settings_df = pd.read_csv(path)
    # Check column names
    assert (list(settings_df.columns) == ['Model type', 'Variant', 'Level', 'Dataset type', 'Micron Resolution', 'Downsampling Method', 'CLR', 'Colour Augmentation', 'Epoch']), 'Settings CSV columns wrongly named'
    # Check model types
    assert (all([x in ['VGG', 'ResNet', 'Inception', 'HistoNet', 'NASNet', 'Xception', 'MobileNet'] for x in settings_df.values[:, 0]])), 'Incorrect model types'
    # Check variants
    for iter_session in range(settings_df.shape[0]):
        if settings_df.values[iter_session, 0] == 'ResNet' and settings_df.values[iter_session, 1] not in \
                ['resnet_18', 'resnet_34', 'resnet_50']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for ResNet'
        elif settings_df.values[iter_session, 0] == 'Inception' and settings_df.values[iter_session, 1] not in ['Default']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for Inception'
        elif settings_df.values[iter_session, 0] == 'HistoNet' and settings_df.values[iter_session, 1] not in ['Series-1.0']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for HistoNet'
    # Check levels
    assert (all([x in ['L1', 'L2', 'L2+', 'L3', 'L3+', 'L3a', 'L3a+'] for x in settings_df.values[:, 2]])), 'Incorrect levels'
    # Check dataset types
    assert (all([x in ['ADP-Release1', 'ADP-Release1-Flat'] for x in settings_df.values[:, 3]])), 'Incorrect dataset types'
    # Check micron resolutions
    assert (all([x in [8, 4, 3, 2, 1] for x in settings_df.values[:, 4]])), 'Incorrect micron resolutions'
    # Check downsampling methods
    assert (all([x in ['bicubic', 'maxpol'] for x in settings_df.values[:, 5]])), 'Incorrect downsampling methods'
    # Check CLRs
    assert (all([x in [True, False] for x in settings_df.values[:, 6]])), 'Incorrect CLRs'
    # Check Colour Augmentations
    assert (all([x in [True, False] for x in settings_df.values[:, 7]])), 'Incorrect Colour Augmentations'
    # Check Epoch integer value
    assert (all([isinstance(x, int) for x in settings_df.values[:, 8]])), 'Incorrect Epoch Value'
    return settings_df

# Limit GPU usage
def set_gpu(gpu_limit):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = gpu_limit
    set_session(tf.Session(config=config))

# Clear model
def clear_model(model):
    del model
    K.clear_session()