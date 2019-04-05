import os
import math
import time
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import numpy as np
from clr_callback import *
import matplotlib.pyplot as plt
import pandas as pd

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
            self.csv_path = os.path.join(self.dataset_dir, 'lbl', 'ADP_EncodedLabels_Release1.csv')
            dataset_type_str = 'Release1'
        elif self.dataset_type == 'ADP-Release1-Flat':
            self.csv_path = os.path.join(self.dataset_dir, 'lbl', 'ADP_EncodedLabels_Release1_Flat.csv')
            dataset_type_str = 'Release1Flat'
        self.class_names, self.num_classes = htt_def.get_htts(self.level, self.dataset_type)

        # Tuning settings (only edit if tuning)
        self.should_run_range_test = False

        # Problem-specific settings
        if self.should_clr:
            clr_str = 'clr'
        else:
            clr_str = '_noclr'
        self.sess_id = 'adp_' + self.model_type + '_' + self.variant + '_' + str(self.level) + '_' + dataset_type_str + '_' + clr_str

        # Path settings (dataset directory)
        self.splits_dir = os.path.join(self.dataset_dir, 'splits')

        # Path settings (current directory)
        cur_path = os.path.abspath(os.path.curdir)
        self.log_dir = os.path.join(cur_path, 'log', self.sess_id)
        self.tmp_dir = os.path.join(cur_path, 'tmp')
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
        # Read splits
        train_inds = np.load(os.path.join(self.splits_dir, 'train.npy'))
        valid_inds = np.load(os.path.join(self.splits_dir, 'valid.npy'))
        test_inds = np.load(os.path.join(self.splits_dir, 'test.npy'))
        # Split dataframe
        train_df = all_df.loc[train_inds, :]
        valid_df = all_df.loc[valid_inds, :]
        test_df = all_df.loc[test_inds, :]
        # Get train class counts
        train_class_counts = [x for i, x in enumerate(np.sum(train_df.values[:, 1:], axis=0)) if train_df.columns[i+1] in self.class_names]

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
                    preprocessing_function=self.normalize)   # normalize by subtracting training set image mean, dividing by training set image std
        train_generator = datagen.flow_from_dataframe(dataframe=train_df,
                                                      directory=self.img_dir,
                                                      x_col='Patch Names',
                                                      y_col=self.class_names,
                                                      batch_size=self.batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]))
        valid_generator = datagen.flow_from_dataframe(dataframe=valid_df,
                                                      directory=self.img_dir,
                                                      x_col='Patch Names',
                                                      y_col=self.class_names,
                                                      batch_size=self.batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]))
        test_generator = datagen.flow_from_dataframe(dataframe=test_df,
                                                      directory=self.img_dir,
                                                      x_col='Patch Names',
                                                      y_col=self.class_names,
                                                      batch_size=self.batch_size,
                                                      class_mode='other',
                                                      target_size=(size[0], size[1]))
        return train_generator, valid_generator, test_generator, train_class_counts

    # Train the model
    def train(self):
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
                clr = CyclicLR(base_lr=0.001, max_lr=0.1, step_size=3 * x_train.shape[0] // self.batch_size)
                self.model.fit(x_train, y_train, callbacks=[clr], epochs=3, class_weight=class_weights)
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
                                callbacks=[lr_reduce_cb, tensorboard_cb],
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