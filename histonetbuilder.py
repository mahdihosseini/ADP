import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras import regularizers

def build_histonet_series_1(input_shape, weight_decay, num_classes, config_code):
    if config_code == 'Series-1.0':
        conv_depths = [32, 32, 128, 256, 128, 256]
        stride_size = [1, 1, 1, 1, 1, 1]
        block_ID = [1, 2, 3, 4, 5, 6]
        kernel_size = [7, 5, 5, 5, 7, 3]
        dropout_vals = []
        pooling_size = [2, 2, 2, 2, 2, 2]
        GMP_activate = True
        fc_depths = [512, num_classes]
    else:
        print('Unsupported config_code!')

    #   configure if global-max-pooling is used at the end of CNN
    if GMP_activate == True:
        N = len(pooling_size)
        pooling_size_array = np.array(pooling_size)
        scaling_factor = input_shape[0] / np.prod(pooling_size_array[0:N - 1])
        stride_size_array=np.array(stride_size)
        scaling_factor = scaling_factor / np.prod(stride_size_array)
        scaling_factor = int(scaling_factor)
        pooling_size[N - 1] = scaling_factor

    #   build CNN architecture series-1 type ==>
    model = Sequential()
    iteration_pooling = 0
    iteration_dropout = 0
    for iteration_layer in range(len(conv_depths)):
        if iteration_layer > 0:
            model.add(
                Conv2D(conv_depths[iteration_layer], kernel_size=(kernel_size[iteration_layer], kernel_size[iteration_layer]), strides=(stride_size[iteration_layer], stride_size[iteration_layer]),
                       padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        else:
            model.add(
                Conv2D(conv_depths[iteration_layer], kernel_size=(kernel_size[iteration_layer], kernel_size[iteration_layer]), strides=(stride_size[iteration_layer], stride_size[iteration_layer]),
                       padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        #   determine whether it is the last layer of each block, so to add maxpooling instead of dropout
        block_ID_current = block_ID[iteration_layer]
        block_ID_last = block_ID.index(block_ID_current) + block_ID.count(block_ID_current) - 1
        if iteration_layer == block_ID_last:
            model.add(MaxPooling2D(pool_size=(pooling_size[iteration_pooling], pooling_size[iteration_pooling])))
            iteration_pooling = iteration_pooling + 1
        else:
            if dropout_vals[iteration_dropout]:
                model.add(Dropout(dropout_vals[iteration_dropout]))
            iteration_dropout = iteration_dropout + 1

    # Simple non-global pooling configurations (flatten + FC)
    model.add(Flatten())

    # Dense Layers
    model.add(Dense(fc_depths[0], kernel_regularizer=regularizers.l2(weight_decay)))
    if len(fc_depths) > 1:
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(fc_depths[1], kernel_regularizer=regularizers.l2(weight_decay)))

    # Add final sigmoid (common to all)
    model.add(Activation('sigmoid'))

    return model

# def build_histonet_series_2(input_shape, weight_decay, num_classes, config_code):
