import pandas as pd
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras.backend as K

# Read settings CSV
def read_settings(path):
    settings_df = pd.read_csv(path)
    # Check column names
    assert (list(settings_df.columns) == ['Model type', 'Variant', 'Level', 'Dataset type', 'CLR']), 'Settings CSV columns wrongly named'
    # Check model types
    assert (all([x in ['VGG', 'ResNet', 'Inception'] for x in settings_df.values[:, 0]])), 'Incorrect model types'
    # Check variants
    for iter_session in range(settings_df.shape[0]):
        if settings_df.values[iter_session, 0] == 'VGG' and settings_df.values[iter_session, 1] not in \
            ['Default', 'V1.2', 'V1.3', 'V1.4', 'V1.5', 'V1.6', 'V1.7', 'V1.8', 'V1.9', 'V1.10', 'V2.1']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for VGG'
        elif settings_df.values[iter_session, 0] == 'ResNet' and settings_df.values[iter_session, 1] not in \
            ['Default', 'ResNet-34', 'ResNet-50', 'ResNet-101', 'ResNet-152']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for ResNet'
        elif settings_df.values[iter_session, 0] == 'Inception' and settings_df.values[iter_session, 1] not in \
            ['Default']:
            assert (False), 'Incorrect variant ' + settings_df.values[iter_session, 1] + ' for Inception'
    # Check levels
    assert (all([x in ['L1', 'L2', 'L2+', 'L3', 'L3+', 'L3a', 'L3a+'] for x in settings_df.values[:, 2]])), 'Incorrect levels'
    # Check dataset types
    assert (all([x in ['ADP-Release1', 'ADP-Release1-Flat'] for x in settings_df.values[:, 3]])), 'Incorrect dataset types'
    # Check CLRs
    assert (all([x in [True, False] for x in settings_df.values[:, 4]])), 'Incorrect CLRs'
    return settings_df

# Limit GPU usage
def set_gpu(gpu_limit):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_limit
    set_session(tf.Session(config=config))

# Clear model
def clear_model(model):
    del model
    K.clear_session()