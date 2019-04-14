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

def get_datagen(csv_path, img_dir, splits_dir, class_names, size):
    all_df = pd.read_csv(csv_path)
    # Read splits
    train_inds = np.load(os.path.join(splits_dir, 'train.npy'))
    valid_inds = np.load(os.path.join(splits_dir, 'valid.npy'))
    test_inds = np.load(os.path.join(splits_dir, 'test.npy'))
    # Split dataframe
    train_df = all_df.loc[train_inds, :]
    valid_df = all_df.loc[valid_inds, :]
    test_df = all_df.loc[test_inds, :]
    # Get train class counts
    train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if
                          train_df.columns[i + 1] in class_names]

    def normalize(x):
        x = (x - 193.09203) / (56.450138 + 1e-7)
        return x

    # Set up data generators
    datagen_aug = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True,  # randomly flip images
        preprocessing_function=normalize)  # normalize by subtracting training set image mean, dividing by training set image std
    datagen_nonaug = ImageDataGenerator(
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
    train_generator = datagen_aug.flow_from_dataframe(dataframe=train_df,
                                                      directory=img_dir,
                                                      x_col='Patch Names',
                                                      y_col=class_names,
                                                      batch_size=16,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=True)
    valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=valid_df,
                                                         directory=img_dir,
                                                         x_col='Patch Names',
                                                         y_col=class_names,
                                                         batch_size=16,
                                                         class_mode='other',
                                                         target_size=(size[0], size[1]),
                                                         shuffle=False)
    test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                        directory=img_dir,
                                                        x_col='Patch Names',
                                                        y_col=class_names,
                                                        batch_size=16,
                                                        class_mode='other',
                                                        target_size=(size[0], size[1]),
                                                        shuffle=False)
    return train_generator, valid_generator, test_generator, train_class_counts

# Find optimal threshold
def find_thresh(tpr, fpr, thresh):
    mode = 'SensEqualsSpec'
    if mode == 'SensEqualsSpec':
        threshold = thresh[np.argmin(abs(tpr-(1-fpr)))]
    return threshold

# Get thresholded class accuracies
def get_thresholded_metrics(target, predictions, thresholds, level, class_names, unaugmented_class_names):

    def threshold_predictions(predictions, thresholds, class_names):
        predictions_thresholded_simple = predictions >= thresholds
        predictions_thresholded_hbr = np.zeros_like(predictions_thresholded_simple)
        for iter_class in range(len(class_names)):
            class_name = class_names[iter_class]
            ancestor_classes = ['.'.join(class_name.split('.')[:i+1]) for i, x in enumerate(class_name.split('.')[:-1])]
            ancestor_class_inds = [i for i, x in enumerate(class_names) if x in ancestor_classes]
            predictions_thresholded_hbr[:, iter_class] = np.logical_and(predictions_thresholded_simple[:, iter_class],
                np.all(predictions_thresholded_simple[:, ancestor_class_inds], axis=1))
        return predictions_thresholded_hbr

    # Obtain thresholded predictions
    if '+' not in level:
        predictions_thresholded = predictions >= thresholds
    else:
        predictions_thresholded = threshold_predictions(predictions, thresholds, class_names)
        # Remove augmented classes and evaluate accuracy
        unaugmented_class_inds = [i for i,x in enumerate(class_names) if x in unaugmented_class_names]
        target = target[:, unaugmented_class_inds]
        predictions_thresholded = predictions_thresholded[:, unaugmented_class_inds]

    # Obtain metrics
    cond_positive = np.sum(target == 1, 0)
    cond_negative = np.sum(target == 0, 0)

    true_positive = np.sum((target == 1) & (predictions_thresholded == 1), 0)
    false_positive = np.sum((target == 0) & (predictions_thresholded == 1), 0)
    true_negative = np.sum((target == 0) & (predictions_thresholded == 0), 0)
    false_negative = np.sum((target == 1) & (predictions_thresholded == 0), 0)

    class_tprs = true_positive / cond_positive
    class_fprs = false_positive / cond_negative
    class_tnrs = true_negative / cond_negative
    class_fnrs = false_negative / cond_positive

    class_accs = np.sum(target == predictions_thresholded, 0) / predictions_thresholded.shape[0]
    class_f1s = (2*true_positive) / (2*true_positive + false_positive + false_negative)

    return class_tprs, class_fprs, class_tnrs, class_fnrs, class_accs, class_f1s

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
    img_dir = os.path.join(DATASET_DIR, 'img-' + str(mdl_ldr.size[0]))
    csv_path = os.path.join(DATASET_DIR, 'lbl', 'ADP_EncodedLabels_Release1.csv')
    train_generator, valid_generator, test_generator, train_class_counts = \
        get_datagen(csv_path, img_dir, SPLITS_DIR, class_names, mdl_ldr.size)
    # Compile and evaluate
    opt = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    # - Get ROC analysis
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
    pred_valid = model.predict_generator(valid_generator, steps=len(valid_generator))
    pred_test = model.predict_generator(test_generator, steps=len(test_generator))
    # test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
    if sess_type == 'old':
        all_df = pd.read_csv(csv_path)
        all_class_names = [x for x in all_df.columns[1:] if '.X' not in x]
        idx = [i for i, x in enumerate(all_class_names) if x in class_names]
        pred_valid = pred_valid[:, np.array(idx)]
        pred_test = pred_test[:, np.array(idx)]
        class_weights = [train_generator.n / np.sum(np.float64(all_df.values[:, 1:]), axis=0)]
    elif sess_type == 'new':
        class_weights = [train_generator.n / np.array(train_class_counts)]
    print('\n'.join(class_names))

    thresh_md = 'Python'
    if thresh_md == 'Python':
        class_thresholds = []
        for iter_class in range(pred_valid.shape[1]):
            fpr, tpr, thresholds = roc_curve(valid_generator.data[:, iter_class], pred_valid[:, iter_class])
            class_thresholds.append(find_thresh(tpr, fpr, thresholds))
            class_thresholds[iter_class] = min(max(class_thresholds[iter_class], 1 / 3), 1)
    elif thresh_md == 'Matlab':
        class_thresholds = [0.5275, 0.4981, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.6568,
                            0.3333, 0.3333, 0.3333, 0.3976, 0.3805, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.8087,
                            0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.7766, 0.3333, 0.3333,
                            0.3333, 0.3333, 0.5181, 0.3333, 0.3333, 0.3333, 0.5016, 0.3333, 0.9592, 0.6495, 0.6315,
                            0.3333, 0.3333]

    class_tprs, class_fprs, class_tnrs, class_fnrs, class_accs, class_f1s = get_thresholded_metrics(test_generator.data,
                                                    pred_test, class_thresholds, level, class_names, unaugmented_class_names)
    a=1