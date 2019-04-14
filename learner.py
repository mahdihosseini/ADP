import os
import math
import time
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from keras import optimizers
import openpyxl
from openpyxl import load_workbook
import numpy as np
from clr_callback import *
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from model_loader import ModelLoader
import htt_def

class Learner:
    # Initialize
    def __init__(self, params):
        self.dataset_dir = params['dataset_dir']
        self.model_type = params['model']
        self.variant = params['variant']
        self.level = params['level']
        self.dataset_type = params['dataset_type']
        self.should_clr = params['should_clr']

        # Hyperparameters
        self.batch_size = 16
        self.epochs = 80
        self.lr_initial = 0.1
        self.lr_decay = 1e-6
        self.lr_drop = 20
        self.momentum = 0.9

        # Level-specific settings
        if self.dataset_type == 'ADP-Release1':
            # self.csv_path = os.path.join(self.dataset_dir, 'lbl', 'ADP_EncodedLabels_Release1.csv')
            self.csv_path = os.path.join(self.dataset_dir, 'lbl', 'ADP_ListLabelsCategorical_Release1.csv')
            dataset_type_str = 'Release1'
        elif self.dataset_type == 'ADP-Release1-Flat':
            self.csv_path = os.path.join(self.dataset_dir, 'lbl', 'ADP_EncodedLabels_Release1_Flat.csv')
            dataset_type_str = 'Release1Flat'
        self.class_names, self.num_classes, self.unaugmented_class_names = htt_def.get_htts(self.level, self.dataset_type)
        self.class_names.sort()
        self.unaugmented_class_names.sort()

        # Tuning settings (only edit if tuning)
        self.should_run_range_test = False

        # Problem-specific settings
        if self.should_clr:
            clr_str = 'clr'
        else:
            clr_str = 'noclr'
        self.sess_id = 'adp_' + self.model_type + '_' + self.variant + '_' + str(self.level) + '_' + dataset_type_str + '_' + clr_str

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

    # Normalize the inputted data to obtain zero-mean, unit-variance distribution
    def normalize(self, x):
        x = (x - 193.09203) / (56.450138 + 1e-7)
        return x

    # Get data generators
    def get_datagen(self, size):
        all_df = pd.read_csv(self.csv_path)
        all_df['Labels'] = all_df['Labels'].apply(lambda x: x.split(';'))

        # Read splits
        train_inds = np.load(os.path.join(self.splits_dir, 'train.npy'))
        valid_inds = np.load(os.path.join(self.splits_dir, 'valid.npy'))
        test_inds = np.load(os.path.join(self.splits_dir, 'test.npy'))
        # Split dataframe
        train_df = all_df.loc[train_inds, :]
        valid_df = all_df.loc[valid_inds, :]
        test_df = all_df.loc[test_inds, :]
        # Get train class counts
        # train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if train_df.columns[i+1] in self.class_names]
        # train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0))]

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
                    preprocessing_function=self.normalize)   # normalize by subtracting training set image mean, dividing by training set image std
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
                    preprocessing_function=self.normalize)  # normalize by subtracting training set image mean, dividing by training set image std
        train_generator = datagen_aug.flow_from_dataframe(dataframe=train_df,
                                                      directory=self.img_dir,
                                                      x_col='Patch Names',
                                                      y_col='Labels',
                                                      batch_size=self.batch_size,
                                                      class_mode='categorical',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=True)
        valid_generator = datagen_nonaug.flow_from_dataframe(dataframe=valid_df,
                                                      directory=self.img_dir,
                                                      x_col='Patch Names',
                                                      y_col='Labels',
                                                      batch_size=self.batch_size,
                                                      class_mode='categorical',
                                                      target_size=(size[0], size[1]),
                                                      shuffle=False)
        test_generator = datagen_nonaug.flow_from_dataframe(dataframe=test_df,
                                                     directory=self.img_dir,
                                                     x_col='Patch Names',
                                                     y_col='Labels',
                                                     batch_size=self.batch_size,
                                                     class_mode='categorical',
                                                     target_size=(size[0], size[1]),
                                                     shuffle=False)
        valid_generator.class_indices = train_generator.class_indices
        # for iter_img in range(valid_generator.n):
        #     for iter_class in range(len(valid_generator.labels[iter_img])):
        #         if valid_generator.labels[iter_img][iter_class] >= 1:
        #             valid_generator.labels[iter_img][iter_class] += 1
        test_generator.class_indices = train_generator.class_indices
        # for iter_img in range(test_generator.n):
        #     for iter_class in range(len(test_generator.labels[iter_img])):
        #         if test_generator.labels[iter_img][iter_class] >= 11 and test_generator.labels[iter_img][iter_class] < 40:
        #             test_generator.labels[iter_img][iter_class] += 1
        #         elif test_generator.labels[iter_img][iter_class] >= 40:
        #             test_generator.labels[iter_img][iter_class] += 2
        flat_list = [item for sublist in train_generator.labels for item in sublist]
        train_class_counts = [flat_list.count(i) for i, x in enumerate(train_generator.class_indices)]
        return train_generator, valid_generator, test_generator, train_class_counts

    # Get data generators
    def get_datagen_old(self, size):
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
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=True,  # randomly flip images
            preprocessing_function=self.normalize)  # normalize by subtracting training set image mean, dividing by training set image std
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
            preprocessing_function=self.normalize)  # normalize by subtracting training set image mean, dividing by training set image std
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
        for iter_class in range(predictions.shape[1]):
            fprs, tprs, thresholds = roc_curve(target[:, iter_class], predictions[:, iter_class])
            opt_thresh = min(max(get_opt_thresh(tprs, fprs, thresholds), thresh_rng[0]), thresh_rng[1])
            class_thresholds.append(opt_thresh)
            class_fprs.append(fprs)
            class_tprs.append(tprs)
        # class_thresholds = [0.5275, 0.4981, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.6568,
        #                     0.3333, 0.3333, 0.3333, 0.3976, 0.3805, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.8087,
        #                     0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.3333, 0.5000, 0.7766, 0.3333, 0.3333,
        #                     0.3333, 0.3333, 0.5181, 0.3333, 0.3333, 0.3333, 0.5016, 0.3333, 0.9592, 0.6495, 0.6315,
        #                     0.3333, 0.3333]
        return class_thresholds, class_fprs, class_tprs

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
            # Remove augmented classes and evaluate accuracy
            unaugmented_class_inds = [i for i,x in enumerate(self.class_names) if x in self.unaugmented_class_names]
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

    def write_to_excel(self, metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s):
        # Find mean metrics
        not_nan_htt_inds = np.argwhere(~np.any(np.isnan(np.vstack((metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s))), axis=0))
        mean_tpr = np.mean(metric_tprs[not_nan_htt_inds])
        mean_fpr = np.mean(metric_fprs[not_nan_htt_inds])
        mean_tnr = np.mean(metric_tnrs[not_nan_htt_inds])
        mean_fnr = np.mean(metric_fnrs[not_nan_htt_inds])
        mean_acc = np.mean(metric_accs[not_nan_htt_inds])
        mean_f1 = np.mean(metric_f1s[not_nan_htt_inds])

        df = pd.DataFrame(
            {'HTT': self.unaugmented_class_names + ['Average'], 'TPR': list(metric_tprs) + [mean_tpr],
             'FPR': list(metric_fprs) + [mean_fpr], 'TNR': list(metric_tnrs) + [mean_tnr],
             'FNR': list(metric_fnrs) + [mean_fnr], 'ACC': list(metric_accs) + [mean_acc],
             'F1': list(metric_f1s) + [mean_f1]},
            columns=['HTT', 'TPR', 'FPR', 'TNR', 'FNR', 'ACC', 'F1'])

        all_xlsx_path = os.path.join(self.eval_dir, 'metrics.xlsx')
        if not os.path.exists(all_xlsx_path ):
            df.to_excel(all_xlsx_path, sheet_name=self.sess_id)
        else:
            book = load_workbook(all_xlsx_path)
            writer = pd.ExcelWriter(all_xlsx_path, engine='openpyxl')
            writer.book = book
            writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
            df.to_excel(writer, sheet_name=self.sess_id)
            writer.save()

        sess_xlsx_path = os.path.join(self.eval_dir, 'metrics_' + self.sess_id + '.xlsx')
        df.to_excel(sess_xlsx_path)

    # Train the model
    def train(self):
        print('Starting training for ' + self.sess_id)
        # Load model
        mdl_ldr = ModelLoader(params={'model': self.model_type, 'variant': self.variant, 'num_classes': self.num_classes})
        self.model = mdl_ldr.load_model()
        self.img_dir = os.path.join(self.dataset_dir, 'img-' + str(mdl_ldr.size[0]))

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
        # def EpMpS_accuracy(y_true, y_pred):
        #     return K.mean(K.all(K.greater_equal(y_pred[:, 0], 0.5), K.greater_equal(y_true[:, 0], 0.5)))
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
                if (iter_clr_sess+1) * self.lr_drop <= self.epochs:
                    curr_epochs = self.lr_drop
                else:
                    curr_epochs = self.epochs - (iter_clr_sess-1) * self.lr_drop
                self.model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n // self.batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid_generator.n // self.batch_size,
                                epochs=curr_epochs,
                                callbacks=[clr, tensorboard_cb],
                                verbose=2,
                                class_weight=class_weights)
                curr_base_lr = init_base_lr * .5 ** iter_clr_sess
                curr_max_lr = init_max_lr * .5 ** iter_clr_sess
                clr._reset(new_base_lr=curr_base_lr, new_max_lr=curr_max_lr, new_step_size=step_sz)
        else:
            self.model.fit_generator(generator=train_generator,
                                steps_per_epoch=train_generator.n // self.batch_size,
                                validation_data=valid_generator,
                                validation_steps=valid_generator.n // self.batch_size,
                                epochs=self.epochs,
                                callbacks=[lr_reduce_cb, tensorboard_cb],
                                verbose=2,
                                class_weight=class_weights)

        # Save trained weights
        print('Saving model weights')
        weights_path = os.path.join(self.model_dir, self.sess_id + '.h5')
        self.model.save_weights(weights_path)

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
        mdl_ldr = ModelLoader(params={'model': self.model_type, 'variant': self.variant, 'num_classes': self.num_classes})
        self.img_dir = os.path.join(self.dataset_dir, 'img-' + str(mdl_ldr.size[0]))
        train_generator, valid_generator, test_generator, train_class_counts = self.get_datagen(mdl_ldr.size)
        valid_data = np.zeros((valid_generator.n, self.num_classes))
        for iter_img in range(valid_generator.n):
            curr_htts = np.array(valid_generator.labels[iter_img])
            curr_htts[curr_htts >= 1] += 1
            valid_data[iter_img, curr_htts] = 1
        test_data = np.zeros((test_generator.n, self.num_classes))
        for iter_img in range(test_generator.n):
            curr_htts = np.array(test_generator.labels[iter_img])
            curr_htts[curr_htts >= 32] += 1
            curr_htts[curr_htts >= 39] += 1
            test_data[iter_img, curr_htts] = 1

        # Compile and evaluate
        opt = optimizers.SGD(lr=self.lr_initial, decay=self.lr_decay, momentum=self.momentum, nesterov=True)
        self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['binary_accuracy'])
        train_generator.shuffle = False
        # pred_train = self.model.predict_generator(train_generator, steps=len(train_generator))
        pred_valid = self.model.predict_generator(valid_generator, steps=len(valid_generator))
        # for iter_img in range(valid_generator.n):
        #     curr_htts = np.array(valid_generator.labels[iter_img])
        #     curr_htts[curr_htts >= 0] += 1
        #     pred_valid[iter_img, curr_htts] = 1
        pred_test = self.model.predict_generator(test_generator, steps=len(test_generator))
        # for iter_img in range(test_generator.n):
        #     curr_htts = np.array(test_generator.labels[iter_img])
        #     curr_htts[curr_htts >= 10] += 1
        #     curr_htts[curr_htts >= 39] += 1
        #     pred_test[iter_img, curr_htts] = 1

        plt.figure(1)
        plt.plot(np.sum(test_data, axis=0))
        plt.plot(np.sum(pred_test, axis=0))
        plt.show()

        # Get ROC analysis
        # - Get optimal class thresholds
        class_thresholds, _, _ = self.get_optimal_thresholds(valid_data, pred_valid)
        # - Get thresholded class accuracies
        metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s = self.get_thresholded_metrics(test_data, pred_test, class_thresholds)
        # - Plot ROC curves
        _, class_fprs, class_tprs = self.get_optimal_thresholds(test_data, pred_test)
        self.plot_rocs(class_fprs, class_tprs)
        # - Write metrics to Excel
        self.write_to_excel(metric_tprs, metric_fprs, metric_tnrs, metric_fnrs, metric_accs, metric_f1s)
        a=1