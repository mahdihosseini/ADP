import pandas as pd

import utilities

from learner import Learner

DATASET_DIR = r'X:\adp-data'    # Directory containing downloaded ADP data
PROCESSING_MODE = 'gpu'         # {'cpu', 'gpu'}
if PROCESSING_MODE == 'gpu':
    GPU_LIMIT = 1
    utilities.set_gpu(GPU_LIMIT)
# Read in settings
settings_df = utilities.read_settings('settings.csv')

# Iterate through session settings
for iter_session in range(settings_df.shape[0]):
    session_settings = settings_df.values[iter_session, :]
    lrn = Learner(params={'dataset_dir': DATASET_DIR, 'model': session_settings[0],
                          'variant': session_settings[1], 'level': session_settings[2],
                          'dataset_type': session_settings[3], 'should_clr': session_settings[4]})
    lrn.train()
    lrn.test()
    if PROCESSING_MODE == 'gpu':
        utilities.clear_model(lrn.model)
    a=1

# lrn = Learner(params={'dataset_dir': DATASET_DIR, 'model': MODELS[iter_session], 'level': LEVELS[iter_session]})
#     models, variants, levels, clrs = read_settings('settings.csv')
a=1