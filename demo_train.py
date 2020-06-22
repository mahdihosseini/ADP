import pandas as pd

import utilities

from learner import Learner

DATASET_DIR = r'G:\database\adp-data\ADP V1.0 Release'

PROCESSING_MODE = 'gpu'         # {'cpu', 'gpu'}
if PROCESSING_MODE == 'gpu':
    GPU_LIMIT = 1
    utilities.set_gpu(GPU_LIMIT)
# Read in settings
settings_df = utilities.read_settings('settings.csv')

# Iterate through session settings
for iter_session in range(settings_df.shape[0]):
    session_settings = settings_df.values[iter_session, :]
    lrn = Learner(params={'dataset_dir': DATASET_DIR,
                          'model': session_settings[0],
                          'variant': session_settings[1],
                          'level': session_settings[2],
                          'dataset_type': session_settings[3],
                          'micron_res': session_settings[4],
                          'downsampling_method': session_settings[5],
                          'should_clr': session_settings[6],
                          'should_colour_augment': session_settings[7],
                          'number_of_epochs': session_settings[8]})
    lrn.train()
    lrn.test()

    if PROCESSING_MODE == 'gpu':
        utilities.clear_model(lrn.model)
    a=1

