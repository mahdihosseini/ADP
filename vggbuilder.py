import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras import regularizers

def build_vgg16(input_shape, weight_decay, num_classes, num_blocks):
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    if num_blocks >= 2:
        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    if num_blocks >= 3:
        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 4
    if num_blocks >= 4:
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 5
    if num_blocks >= 5:
        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

    # Fully Connected
    model.add(Flatten())
    model.add(Dense(256, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(num_classes, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('sigmoid'))

    return model

def build_vgg16_V1_variant(input_shape, weight_decay, num_classes, config_code):
    # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.
    model = Sequential()

    if config_code == 'V1.2':
        conv_depths = [32, 32, 64, 64, 128, 128, 128]
        fc_depths = [512, num_classes]
    elif config_code == 'V1.3':
        conv_depths = [64, 64, 128, 128, 256, 256, 256]
        fc_depths = [num_classes]
    elif config_code == 'V1.4':
        conv_depths = [32, 32, 64, 64, 128, 128, 128]
        fc_depths = [num_classes]
    elif config_code in ['V1.5', 'V1.6', 'V1.9']:
        conv_depths = [64, 64, 128, 128, 256, 256, num_classes]
    elif config_code in ['V1.7', 'V1.8', 'V1.10']:
        conv_depths = [64, 64, 128, 128, 256, 256, 256]
        fc_depths = [num_classes]
    else:
        print('Unsupported config_code!')

    # Block 1
    model.add(Conv2D(conv_depths[0], (3, 3), padding='same',
                     input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(conv_depths[1], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(conv_depths[2], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[3], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 3
    model.add(Conv2D(conv_depths[4], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[5], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[6], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Simple non-global pooling configurations (flatten + FC)
    if config_code not in ['V1.5', 'V1.6', 'V1.7', 'V1.8']:
        model.add(Flatten())

        # Dense Layers
        model.add(Dense(fc_depths[0], kernel_regularizer=regularizers.l2(weight_decay)))
        if len(fc_depths) > 1:
            model.add(Activation('relu'))
            model.add(BatchNormalization())

            model.add(Dropout(0.5))
            model.add(Dense(fc_depths[1], kernel_regularizer=regularizers.l2(weight_decay)))
    # Global pooling configurations
    else:
        # Global pooling layers
        if config_code in ['V1.5', 'V1.7']:
            model.add(MaxPooling2D(pool_size=(28, 28)))
            model.add(Flatten())
        elif config_code in ['V1.6', 'V1.8']:
            model.add(GlobalAveragePooling2D())
        # Optional FC layers
        if config_code in ['V1.7', 'V1.8']:
            model.add(Dense(fc_depths[0], kernel_regularizer=regularizers.l2(weight_decay)))
            if len(fc_depths) > 1:
                model.add(Activation('relu'))
                model.add(BatchNormalization())

                model.add(Dropout(0.5))
                model.add(Dense(fc_depths[1], kernel_regularizer=regularizers.l2(weight_decay)))
    # Add final sigmoid (common to all)
    model.add(Activation('sigmoid'))

    return model

def build_vgg16_V2_variant(input_shape, weight_decay, num_classes, config_code):
    model = Sequential()

    if config_code == 'V2.1':
        conv_depths = [32, 32, 64, 64, 128, 128, 128]
        fc_depths = [num_classes]
    else:
        print('Unsupported config_code!')

    # Block 1
    model.add(Conv2D(conv_depths[0], (3, 3), padding='same', input_shape=input_shape, kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(conv_depths[1], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(conv_depths[2], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Block 2
    model.add(Conv2D(conv_depths[3], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[4], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[5], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(conv_depths[6], (3, 3), padding='same', kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(MaxPooling2D(pool_size=(56, 56)))
    model.add(Flatten())
    model.add(Dense(fc_depths[0], kernel_regularizer=regularizers.l2(weight_decay)))

    if len(fc_depths) > 1:
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(fc_depths[1], kernel_regularizer=regularizers.l2(weight_decay)))
    # Add final sigmoid (common to all)
    model.add(Activation('sigmoid'))

    return model