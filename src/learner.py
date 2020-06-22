import os
import math
import time
import keras
import keras.backend as K
import scipy
from scipy.stats import circmean
from scipy.stats import circstd
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers
import openpyxl
from openpyxl import load_workbook
import numpy as np
from src.clr_callback import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import matplotlib.pylab as plt
import matplotlib
import time
from src.model_loader import ModelLoader
import src.htt_def as htt_def
import skimage

class Learner:
    # Initialize
    def __init__(self, params):
        self.dataset_dir = params['dataset_dir']
        self.model_type = params['model']
        self.variant = params['variant']
        self.level = params['level']
        self.dataset_type = params['dataset_type']
        self.micron_res = params['micron_res']
        self.downsampling_method = params['downsampling_method']
        self.should_clr = params['should_clr']
        self.should_colour_augment = params['should_colour_augment']

        # Hyperparameters
        self.batch_size = 32
        self.epochs = params['number_of_epochs']
        self.lr_initial = 0.1
        self.lr_decay = 1e-6
        self.lr_drop = 20
        self.momentum = 0.9

        # Level-specific settings
        if self.dataset_type == 'ADP-Release1':
            self.csv_path = os.path.join(self.dataset_dir, 'ADP_EncodedLabels_Release1.csv')
            dataset_type_str = 'Release1'
        elif self.dataset_type == 'ADP-Release1-Flat':
            self.csv_path = os.path.join(self.dataset_dir, 'ADP_EncodedLabels_Release1_Flat.csv')
            dataset_type_str = 'Release1Flat'
        self.class_names, self.num_classes, self.unaugmented_class_names = htt_def.get_htts(self.level, self.dataset_type)
        # self.class_names.sort()
        # self.unaugmented_class_names.sort()

        # Tuning settings (only edit if tuning)
        self.should_run_range_test = False

        # Problem-specific settings
        if self.should_clr:
            clr_str = 'clr'
        else:
            clr_str = 'noclr'
        if self.should_colour_augment:
            aug_str = 'caug'
        else:
            aug_str = 'nocaug'
        self.sess_id = 'adp_' + self.model_type + '_' + self.variant + '_' + str(self.level) + '_' + \
                       dataset_type_str + '_' + str(self.micron_res) + 'um_' + self.downsampling_method + '_' + clr_str + \
                       aug_str + '_epochs_' + str(self.epochs)

        # Path settings (dataset directory)
        self.splits_dir = os.path.join(self.dataset_dir, 'splits')

        # Path settings (current directory)
        cur_path = os.path.abspath(os.path.curdir)
        self.log_dir = os.path.join(cur_path, 'log', self.sess_id)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.tmp_dir = os.path.join(cur_path, 'tmp')
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.eval_dir = os.path.join(cur_path, 'eval')
        if not os.path.exists(self.eval_dir):
            os.makedirs(self.eval_dir)
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)
        self.model_dir = os.path.join(cur_path, 'trained-models')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.ckpt_dir = os.path.join(cur_path, 'ckpt')
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    # Normalize the inputted data to obtain zero-mean, unit-variance distribution
    def normalize(self, x):
        x = (x - 193.09203) / (56.450138 + 1e-7)
        return x

    def custom_augmentation_ycbcr(self, image):
        # Assumes input image is in RGB color space, and returns image in RGB space
        scale_Cb = 1.0
        scale_Cr = 1.0

        A = np.array([[65.481, 128.553, 24.966], [-37.797, -74.203, 112.0], [112.0, -93.786, -18.214]])
        A = A / 255.
        b = np.array([16, 128, 128])

        x = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
        image_ycbcr = x @ A.T + b
        image_ycbcr = image_ycbcr.reshape((image.shape[0], image.shape[1], image.shape[2]))

        Y = image_ycbcr[:, :, 0]
        Cb = image_ycbcr[:, :, 1]
        Cr = image_ycbcr[:, :, 2]

        mean_Cb = np.mean(Cb, axis=(0, 1))
        mean_Cr = np.mean(Cr, axis=(0, 1))
        std_Cb = np.std(Cb, axis=(0, 1))
        std_Cr = np.std(Cr, axis=(0, 1))

        Cb_centered = Cb - mean_Cb
        Cr_centered = Cr - mean_Cr

        Cb_centered_augmented = Cb_centered + np.random.normal(loc=0, scale=(scale_Cb * std_Cb))
        Cr_centered_augmented = Cr_centered + np.random.normal(loc=0, scale=(scale_Cr * std_Cr))

        Cb_augmented = Cb_centered_augmented + mean_Cb
        Cr_augmented = Cr_centered_augmented + mean_Cr

        image_perturbed_ycbcr = np.empty(image.shape)
        image_perturbed_ycbcr[:, :, 0] = Y
        image_perturbed_ycbcr[:, :, 1] = Cb_augmented
        image_perturbed_ycbcr[:, :, 2] = Cr_augmented

        inv_A = np.linalg.inv(A)
        image_perturbed = (image_perturbed_ycbcr - b) @ inv_A.T
        image_perturbed = np.rint(np.clip(image_perturbed, 0, 255)).astype('uint8')
        image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7)
        return image_perturbed

    def custom_augmentation_HSV(self, image):
        # Assumes input image is in RGB color space, and returns image in RGB space
        H_Scale = 1.0
        S_Scale = 1.0

        HSV_image = skimage.color.rgb2hsv(image)
        H = HSV_image[:, :, 0]
        H_rad = H * [2 * math.pi] - math.pi
        S = HSV_image[:, :, 1]
        V = HSV_image[:, :, 2]

        mean_H_rad = circmean(H_rad)
        std_H_rad = circstd(H_rad)
        mean_S = np.mean(S, axis=(0, 1))
        std_S = np.std(S, axis=(0, 1))

        H_rad_centered = np.angle(np.exp(1j * (H_rad - mean_H_rad)))
        H_rad_centered_augmented = H_rad_centered + np.random.normal(loc=0, scale=(H_Scale * std_H_rad))
        H_rad_augmented = np.angle(np.exp(1j * (H_rad_centered_augmented + mean_H_rad)))
        H_augmented = np.divide(H_rad_augmented + math.pi, 2 * math.pi)

        S_centered = S - mean_S
        S_centered_augmented = S_centered + np.random.normal(loc=0, scale=(S_Scale * std_S))
        S_augmented = S_centered_augmented + mean_S

        image_perturbed_HSV = np.empty(image.shape)
        image_perturbed_HSV[:, :, 0] = H_augmented
        image_perturbed_HSV[:, :, 1] = S_augmented
        image_perturbed_HSV[:, :, 2] = V

        image_perturbed = skimage.color.hsv2rgb(image_perturbed_HSV)
        image_perturbed = np.rint(np.clip(image_perturbed, 0, 255)).astype('uint8')
        image_perturbed = (image_perturbed - 193.09203) / (56.450138 + 1e-7)
        return image_perturbed

    # Get data generators
    def get_datagen(self, size):
        all_df = pd.read_csv(self.csv_path)
        # Read splits
        train_inds = np.load(os.path.join(self.splits_dir, 'train.npy'))
        valid_inds = np.load(os.path.join(self.splits_dir, 'valid.npy'))
        test_inds = np.load(os.path.join(self.splits_dir, 'test.npy'))
        # Split dataframe
        train_df = all_df.loc[train_inds, :]
        valid_df = all_df.loc[valid_inds, :]
        test_df = all_df.loc[test_inds, :]


        # Get train class counts
        train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if
                              train_df.columns[i + 1] in self.class_names]
        # train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0))]

        # Set up data generators
        datagen_aug = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            preprocessing_function=self.custom_augmentation_ycbcr)
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
            preprocessing_function=self.normalize)
        train_generator = datagen_aug.flow_from_dataframe(dataframe=train_df,
                                                          directory=self.img_dir,
                                                          x_col='Patch Names',
                                                          y_col=self.class_names,
                                                          batch_size=self.batch_size,
                                                          class_mode='other',
                                                          target_size=(size[0], size[1]),
                                                          shuffle=True)
        valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=valid_df,
                                                             directory=self.img_dir,
                                                             x_col='Patch Names',
                                                             y_col=self.class_names,
                                                             batch_size=self.batch_size,
                                                             class_mode='other',
                                                             target_size=(size[0], size[1]),
                                                             shuffle=False)
        test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                            directory=self.img_dir,
                                                            x_col='Patch Names',
                                                            y_col=self.class_names,
                                                            batch_size=self.batch_size,
                                                            class_mode='other',
                                                            target_size=(size[0], size[1]),
                                                            shuffle=False)
        return train_generator, valid_generator, test_generator, train_class_counts

    # Get optimal thresholds, using ROC analysis
    def get_optimal_thresholds(self, target, predictions, thresh_rng=[1/3, 1]):

        def get_opt_thresh(tprs, fprs, threshs):
            mode = 'SensEqualsSpec'
            if mode == 'SensEqualsSpec':
                opt_thresh = threshs[np.argmin(abs(tprs - (1 - fprs)))]
            return opt_thresh

        class_thresholds = []
        class_fprs = []
        class_tprs = []
        auc_measures = []
        for iter_class in range(predictions.shape[1]):
            fprs, tprs, thresholds = roc_curve(target[:, iter_class], predictions[:, iter_class])
            auc_measure = auc(fprs, tprs)
            opt_thresh = min(max(get_opt_thresh(tprs, fprs, thresholds), thresh_rng[0]), thresh_rng[1])
            class_thresholds.append(opt_thresh)
            class_fprs.append(fprs)
            class_tprs.append(tprs)
            auc_measures.append(auc_measure)
        auc_measures.append(sum(np.sum(target, 0) * auc_measures)/np.sum(target))
        return class_thresholds, class_fprs, class_tprs, auc_measures

    # Get thresholded class accuracies
    def get_thresholded_metrics(self, target, predictions, thresholds):

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
        if '+' not in self.level:
            predictions_thresholded = predictions >= thresholds
        else:
            predictions_thresholded = threshold_predictions(predictions, thresholds, self.class_names)

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
        class_f1s = (2 * true_positive) / (2 * true_positive + false_positive + false_negative)
        #
        cond_positive_T = np.sum(target == 1)
        cond_negative_T = np.sum(target == 0)
        true_positive_T = np.sum((target == 1) & (predictions_thresholded == 1))
        false_positive_T = np.sum((target == 0) & (predictions_thresholded == 1))
        true_negative_T = np.sum((target == 0) & (predictions_thresholded == 0))
        false_negative_T = np.sum((target == 1) & (predictions_thresholded == 0))
        tpr_T = true_positive_T / cond_positive_T
        fpr_T = false_positive_T / cond_negative_T
        tnr_T = true_negative_T / cond_negative_T
        fnr_T = false_negative_T / cond_positive_T
        acc_T = np.sum(target == predictions_thresholded) / np.prod(predictions_thresholded.shape)
        f1_T = (2 * true_positive_T) / (2 * true_positive_T + false_positive_T + false_negative_T)
        #
        class_tprs = np.append(class_tprs, tpr_T)
        class_fprs = np.append(class_fprs, fpr_T)
        class_tnrs = np.append(class_tnrs, tnr_T)
        class_fnrs = np.append(class_fnrs, fnr_T)
        class_accs = np.append(class_accs, acc_T)
        class_f1s  = np.append(class_f1s, f1_T)
        #
        unaugmented_class_inds = [i for i, x in enumerate(self.class_names) if x in self.unaugmented_class_names]
        cond_positive_TU = np.sum(target[:, unaugmented_class_inds] == 1)
        cond_negative_TU = np.sum(target[:, unaugmented_class_inds] == 0)
        true_positive_TU = np.sum((target[:, unaugmented_class_inds] == 1) & (predictions_thresholded[:, unaugmented_class_inds] == 1))
        false_positive_TU = np.sum((target[:, unaugmented_class_inds] == 0) & (predictions_thresholded[:, unaugmented_class_inds] == 1))
        true_negative_TU = np.sum((target[:, unaugmented_class_inds] == 0) & (predictions_thresholded[:, unaugmented_class_inds] == 0))
        false_negative_TU = np.sum((target[:, unaugmented_class_inds] == 1) & (predictions_thresholded[:, unaugmented_class_inds] == 0))
        tpr_TU = true_positive_TU / cond_positive_TU
        fpr_TU = false_positive_TU / cond_negative_TU
        tnr_TU = true_negative_TU / cond_negative_TU
        fnr_TU = false_negative_TU / cond_positive_TU
        acc_TU = np.sum(target[:, unaugmented_class_inds] == predictions_thresholded[:, unaugmented_class_inds]) / np.prod(predictions_thresholded[:, unaugmented_class_inds].shape)
        f1_TU = (2 * true_positive_TU) / (2 * true_positive_TU + false_positive_TU + false_negative_TU)
        #
        class_tprs_U = np.append(class_tprs[unaugmented_class_inds], tpr_TU)
        class_fprs_U = np.append(class_fprs[unaugmented_class_inds], fpr_TU)
        class_tnrs_U = np.append(class_tnrs[unaugmented_class_inds], tnr_TU)
        class_fnrs_U = np.append(class_fnrs[unaugmented_class_inds], fnr_TU)
        class_accs_U = np.append(class_accs[unaugmented_class_inds], acc_TU)
        class_f1s_U  = np.append(class_f1s[unaugmented_class_inds] , f1_TU )

        return class_tprs, class_fprs, class_tnrs, class_fnrs, class_accs, class_f1s, class_tprs_U, class_fprs_U, class_tnrs_U, class_fnrs_U, class_accs_U, class_f1s_U, unaugmented_class_inds

    def plot_rocs(self, class_fprs, class_tprs):
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        for iter_class in range(len(self.unaugmented_class_names)):
            plt.plot(class_fprs[iter_class], class_tprs[iter_class], label=self.unaugmented_class_names[iter_class])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        # plt.show()
        plt.savefig(os.path.join(self.eval_dir, 'ROC_' + self.sess_id + '.png'), bbox_inches='tight')

    def write_to_excel(self, metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s, auc_measures, metric_tprs_U, metric_fprs_U, metric_tnrs_U, metric_fnrs_U, metric_accs_U, metric_f1s_U, auc_measures_U):
        # Start a new Excel
        sess_xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.sess_id + '_all.xlsx')

        df = pd.DataFrame({'HTT': self.class_names + ['Average'],
                           'TPR': list(metric_tprs),
                           'FPR': list(metric_fprs),
                           'TNR': list(metric_tnrs),
                           'FNR': list(metric_fnrs),
                           'ACC': list(metric_accs),
                           'F1': list(metric_f1s),
                           'AUC': auc_measures}, columns=['HTT', 'TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'F1', 'AUC'])
        df.to_excel(sess_xlsx_path)

        # Unaugmented classes
        sess_xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.sess_id + '_unaugmented.xlsx')

        df = pd.DataFrame({'HTT': self.unaugmented_class_names + ['Average'],
                           'TPR': list(metric_tprs_U),
                           'FPR': list(metric_fprs_U),
                           'TNR': list(metric_tnrs_U),
                           'FNR': list(metric_fnrs_U),
                           'ACC': list(metric_accs_U),
                           'F1': list(metric_f1s_U),
                           'AUC': auc_measures_U}, columns=['HTT', 'TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'F1', 'AUC'])
        df.to_excel(sess_xlsx_path)
        a=1

    # Train the model
    def train(self):
        print('Starting training for ' + self.sess_id)
        # Load model
        self.img_size = round(272 / self.micron_res)
        mdl_ldr = ModelLoader(params={'model': self.model_type, 'variant': self.variant, 'num_classes': self.num_classes,
                                      'img_size': self.img_size})
        self.model = mdl_ldr.load_model()
        self.img_dir = os.path.join(self.dataset_dir, 'img_res_' + str(self.micron_res) + 'um_' +
                                    self.downsampling_method)
        assert(os.path.exists(self.img_dir)), 'Image directory not found'

        # Set up callbacks
        def lr_scheduler(epoch):
            return self.lr_initial * (0.5 ** (epoch // self.lr_drop))
        lr_reduce_cb = keras.callbacks.LearningRateScheduler(lr_scheduler)
        tensorboard_cb = keras.callbacks.TensorBoard(log_dir=self.log_dir, write_graph=True)

        # Set up data generators
        print('Setting up data generators')
        train_generator, valid_generator, test_generator, train_class_counts = self.get_datagen(mdl_ldr.size)

        # Compiling model
        print('Compiling model')
        opt = optimizers.SGD(lr=self.lr_initial, decay=self.lr_decay, momentum=self.momentum, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])

        # Save model architecture
        print('Saving model architecture')
        arch_path = os.path.join(self.model_dir, self.sess_id + '.json')

        with open(arch_path, 'w') as f:
            f.write(self.model.to_json())

        # Start training
        class_weights = [train_generator.n / np.array(train_class_counts)]
        if self.should_clr:
            if self.should_run_range_test:
                clr = CyclicLR(base_lr=0.001, max_lr=0.1, step_size=3 * train_generator.n // self.batch_size)
                self.model.fit_generator(generator=train_generator, steps_per_epoch=train_generator.n // self.batch_size,
                                         epochs=3,
                                         callbacks=[clr],
                                         class_weight=class_weights)
                plt.xlabel('Learning Rate')
                plt.ylabel('Binary Accuracy')
                plt.title("LR Range Test")
                plt.plot(clr.history['lr'], clr.history['binary_accuracy'])
                plt.show()

            init_base_lr = 0.001
            init_max_lr = 0.02
            epochs_per_step = 5
            step_sz = epochs_per_step * train_generator.n // self.batch_size
            clr = CyclicLR(base_lr=init_base_lr, max_lr=init_max_lr, step_size=step_sz)
            for iter_clr_sess in range(math.ceil(self.epochs / self.lr_drop)):
                filepath = os.path.join(self.ckpt_dir, self.variant + '_clr_sess_' + iter_clr_sess.__str__() + \
                                        '_Epoch_{epoch:02d}_Val_Loss_{val_loss:.3f}.hdf5')
                checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                                             save_best_only=True, save_weights_only=False,
                                                             mode='max', period=self.lr_drop)
                if (iter_clr_sess+1) * self.lr_drop <= self.epochs:
                    curr_epochs = self.lr_drop
                else:
                    curr_epochs = self.epochs - (iter_clr_sess-1) * self.lr_drop
                self.model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n // self.batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid_generator.n // self.batch_size,
                                epochs=curr_epochs,
                                callbacks=[clr, tensorboard_cb, checkpoint],
                                verbose=2,
                                class_weight=class_weights)
                curr_base_lr = init_base_lr * .5 ** iter_clr_sess
                curr_max_lr = init_max_lr * .5 ** iter_clr_sess
                clr._reset(new_base_lr=curr_base_lr, new_max_lr=curr_max_lr, new_step_size=step_sz)
        else:
            filepath = os.path.join(self.ckpt_dir, self.variant + '_noclr_sess_' + \
                                    'Epoch_{epoch:02d}_Val_Loss_{val_loss:.3f}.hdf5')
            checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                                         save_best_only=True, save_weights_only=False,
                                                         mode='max', period=self.lr_drop)
            self.model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n // self.batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid_generator.n // self.batch_size,
                                epochs=self.epochs,
                                callbacks=[lr_reduce_cb, tensorboard_cb, checkpoint],
                                verbose=2,
                                class_weight=class_weights)

        # Save trained weights
        print('Saving model weights')
        weights_path = os.path.join(self.model_dir, self.sess_id + '.h5')

        self.model.save_weights(weights_path)


    def get_crops2(self, im, overlap, size_ori, size_pat):  # size: size of crop
        '''return numpy array containing crops'''

        n = 0
        cur = 0
        while cur + size_pat <= size_ori:
            cur += size_pat
            n += 1
        n += 1

        step = int(size_pat - size_pat * overlap)
        arr = list()
        for i in range(n):
            for j in range(n):
                start_x = step * i
                start_y = step * j
                if i == n - 1:
                    start_x = size_ori - size_pat
                if j == n - 1:
                    start_y = size_ori - size_pat
                img = im[start_x:(start_x + size_pat), start_y:(start_y + size_pat), :]
                arr.append(img)
        X = np.array(arr)
        return X, n


    # Test the model
    def test(self):
        print('Starting testing for ' + self.sess_id)

        # Load json and create model
        arch_path = os.path.join(self.model_dir, self.sess_id + '.json')

        with open(arch_path, 'r') as f:
            loaded_model_json = f.read()
        self.model = model_from_json(loaded_model_json)

        # Load weights into new model
        weights_path = os.path.join(self.model_dir, self.sess_id + '.h5')

        self.model.load_weights(weights_path)

        # Set up data generator
        self.img_size = round(272 / self.micron_res)
        mdl_ldr = ModelLoader(params={'model': self.model_type, 'variant': self.variant, 'num_classes': self.num_classes, 'img_size': self.img_size})
        self.img_dir = os.path.join(self.dataset_dir, 'img_res_' + str(self.micron_res) + 'um_' + self.downsampling_method)

        train_generator, valid_generator, test_generator, train_class_counts = self.get_datagen(mdl_ldr.size)

        # Compile and evaluate
        opt = optimizers.SGD(lr=self.lr_initial, decay=self.lr_decay, momentum=self.momentum, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        train_generator.shuffle = False
        pred_valid = self.model.predict_generator(valid_generator, steps=len(valid_generator))
        pred_test = self.model.predict_generator(test_generator, steps=len(test_generator))

        # plt.figure(1)
        # plt.plot(np.sum(test_data, axis=0))
        # plt.plot(np.sum(pred_test, axis=0))
        # plt.show()

        # Get ROC analysis
        # - Get optimal class thresholds
        class_thresholds, _, _, _ = self.get_optimal_thresholds(valid_generator.labels, pred_valid)

        print('Saving MATLAB files: Class names and Optimum thresholds')
        x = {}
        x['class_thresholds'] = class_thresholds
        x['class_names'] = self.class_names

        MATLAB_file_path = os.path.join(self.model_dir, self.sess_id + '.mat')

        scipy.io.savemat(MATLAB_file_path, x)

        # - Get thresholded class accuracies
        metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s, metric_tprs_U, metric_fprs_U, metric_tnrs_U, metric_fnrs_U, metric_accs_U, metric_f1s_U, unaugmented_class_inds = self.get_thresholded_metrics(test_generator.labels, pred_test, class_thresholds)
        # - Plot ROC curves
        class_thresholds, class_fprs, class_tprs, auc_measures = self.get_optimal_thresholds(test_generator.labels, pred_test)
        auc_measures_U = [ auc_measures[i] for i in unaugmented_class_inds]
        auc_measures_U.append(auc_measures[-1])
        self.plot_rocs(class_fprs, class_tprs)
        # - Write metrics to Excel
        self.write_to_excel(metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s, auc_measures, metric_tprs_U, metric_fprs_U, metric_tnrs_U, metric_fnrs_U, metric_accs_U, metric_f1s_U, auc_measures_U)
        a=1

    def shuffle_together2(self, a, b):
        perm = np.arange(len(b))
        np.random.shuffle(perm)
        a = a[perm]
        b = b[perm]
        # print("checking shape")
        # print(a.shape)
        # print(b.shape)
        return a, b