import os
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from model_loader import ModelLoader
import htt_def
import cv2

MODELS = ['VGG', 'VGG']
# VARIANTS = ['V1.7', 'V1.7']
VARIANTS = ['Default', 'Default']
LEVELS = ['L3+', 'L3+']
# SESS_IDS = ['atlas_vgg_3b_v1.7_clrdecay_5', 'adp_VGG_V1.7_L3+_Release1_clr']
SESS_IDS = ['atlas_vgg_clrdecay_5', 'adp_VGG_Default_L3+_Release1_clr']
SESS_TYPES = ['old', 'new']
DATASET_TYPES = ['ADP-Release1', 'ADP-Release1']

MODEL_DIR = r'C:\Users\chanlynd\Documents\GitHub\Atlas_of_Digital_Pathology\trained-models\tmp'
DATASET_DIR = r'X:\adp-data'
SPLITS_DIR = r'X:\adp-data\splits'


def normalize(x):
    x = (x - 193.09203) / (56.450138 + 1e-7)
    return x

def get_datagen(csv_path, img_dir, class_names, size):
    test_df = pd.read_csv(csv_path)

    # Set up data generators
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        preprocessing_function=normalize)  # normalize by subtracting training set image mean, dividing by training set image std
    test_generator = datagen.flow_from_dataframe(dataframe=test_df,
                                                 directory=img_dir,
                                                 x_col='Patch Names',
                                                 y_col=class_names,
                                                 batch_size=16,
                                                 class_mode='other',
                                                 target_size=(size[0], size[1]),
                                                 shuffle=False)
    return test_generator

for iter_sess in range(2):
    model_type = MODELS[iter_sess]
    variant = VARIANTS[iter_sess]
    level = LEVELS[iter_sess]
    sess_id = SESS_IDS[iter_sess]
    sess_type = SESS_TYPES[iter_sess]
    dataset_type = DATASET_TYPES[iter_sess]

    class_names, num_classes, unaugmented_class_names = htt_def.get_htts(level, dataset_type)

    # Load json and create model
    arch_path = os.path.join(MODEL_DIR, sess_id + '.json')
    with open(arch_path, 'r') as f:
        loaded_model_json = f.read()
    model = model_from_json(loaded_model_json)

    # Load weights into new model
    weights_path = os.path.join(MODEL_DIR, sess_id + '.h5')
    model.load_weights(weights_path)

    # Set up data generator
    mdl_ldr = ModelLoader(params={'model': model_type, 'variant': variant, 'num_classes': num_classes})
    img_dir = os.path.join(DATASET_DIR, 'dbg')
    csv_path = os.path.join(DATASET_DIR, 'lbl', 'dbg_lbl.csv')
    all_df = pd.read_csv(csv_path)

    test_generator = get_datagen(csv_path, img_dir, class_names, mdl_ldr.size)

    # Compile and evaluate
    opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # - Get ROC analysis
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    pred_test_1 = model.predict_generator(test_generator, steps=1)

    x = cv2.cvtColor(cv2.imread(os.path.join(img_dir, all_df.values[0, 0])), cv2.COLOR_BGR2RGB)
    x = np.expand_dims(x, axis=0)
    x = normalize(x)
    pred_test_2 = model.predict(x)

    # model.predict_generator(ImageDataGenerator.flow(x_train, y_train, batch_size=self.batch_size),
    #                     steps_per_epoch=x_train.shape[0] // self.batch_size,
    #                     epochs=curr_epochs,
    #                     validation_data=(x_valid, y_valid),
    #                     callbacks=[tensorboard_callbeck, clr],
    #                     verbose=2, class_weight=class_weights)

    if sess_type == 'old':
        all_class_names = [x for x in all_df.columns[1:] if '.X' not in x]
        idx = [i for i, x in enumerate(all_class_names) if x in class_names]
        pred_test_1 = pred_test_1[:, np.array(idx)]
        pred_test_2 = pred_test_2[:, np.array(idx)]

    a=1