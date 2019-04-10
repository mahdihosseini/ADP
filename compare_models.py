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
    datagen = ImageDataGenerator(
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
    train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                  directory=img_dir,
                                                  x_col='Patch Names',
                                                  y_col=class_names,
                                                  batch_size=16,
                                                  class_mode='other',
                                                  target_size=(size[0], size[1]))
    valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                                  directory=img_dir,
                                                  x_col='Patch Names',
                                                  y_col=class_names,
                                                  batch_size=16,
                                                  class_mode='other',
                                                  target_size=(size[0], size[1]))
    test_generator = datagen.flow_from_dataframe(dataframe=test_df,
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

for iter_sess in range(2):
    model_type = MODELS[iter_sess]
    variant = VARIANTS[iter_sess]
    level = LEVELS[iter_sess]
    sess_id = SESS_IDS[iter_sess]
    sess_type = SESS_TYPES[iter_sess]
    dataset_type = DATASET_TYPES[iter_sess]

    class_names, num_classes = htt_def.get_htts(level, dataset_type)

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
    pred_test = model.predict_generator(test_generator, steps=len(test_generator))
    # test_loss, test_acc = model.evaluate_generator(test_generator, steps=len(test_generator))
    if sess_type == 'old':
        all_df = pd.read_csv(csv_path)
        all_class_names = [x for x in all_df.columns[1:] if '.X' not in x]
        idx = [i for i, x in enumerate(all_class_names) if x in class_names]
        pred_test = pred_test[:, np.array(idx)]
        class_weights = [train_generator.n / np.sum(np.float64(all_df.values[:, 1:]), axis=0)]
    elif sess_type == 'new':
        class_weights = [train_generator.n / np.array(train_class_counts)]
    print('\n'.join(class_names))
    print(class_weights)
    plt.figure(iter_sess+1)
    plt.plot([0, 1], [0, 1], 'k--')
    class_accuracies = []
    class_thresholds = []
    for iter_class in range(pred_test.shape[1]):
        fpr, tpr, thresholds = roc_curve(test_generator.data[:, iter_class], pred_test[:, iter_class])
        class_thresholds.append(find_thresh(tpr, fpr, thresholds))
        class_thresholds[iter_class] = min(max(class_thresholds[iter_class], 1 / 3), 1)
        class_accuracies.append(np.sum(np.equal(test_generator.data[:, iter_class],
                                                pred_test[:, iter_class] > class_thresholds[
                                                    iter_class])) / test_generator.n)
        # auc_metric = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='HTT code: ' + class_names[iter_class])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    print(class_accuracies)
    print(np.mean(np.array(class_accuracies)))

    # iter_test = 94
    # print(test_generator.filenames[iter_test])
    # print('Target: ' + ', '.join([x for i, x in enumerate(class_names) if test_generator.data[iter_test, i]]))
    # print('Pred: ' + ', '.join([x for i, x in enumerate(class_names) if np.uint8(pred_test[iter_test, i] > class_thresholds[i])]))
    a=1
