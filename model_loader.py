import keras
import vggbuilder
import resnetbuilder
from keras.applications import InceptionV3

from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras import regularizers

class ModelLoader:
    # Initialize
    def __init__(self, params):
        self.model_type = params['model']
        self.variant = params['variant']
        self.num_classes = params['num_classes']

        if self.model_type == 'VGG' or self.model_type == 'ResNet':
            self.size = [224, 224, 3]
        elif self.model_type == 'Inception':
            self.size = [229, 229, 3]
        self.weight_decay = 5e-4

    # Load and customize default Keras networks
    def load_model(self):
        if self.model_type == 'VGG':
            if self.variant == 'Default':
                model = vggbuilder.build_vgg16(input_shape=self.size, weight_decay=self.weight_decay, \
                                               num_classes=self.num_classes, num_blocks=5)
            else:
                if 'V1.' in self.variant:
                    model = vggbuilder.build_vgg16_V1_variant(input_shape=self.size, weight_decay=self.weight_decay, \
                                                          num_classes=self.num_classes, config_code=self.variant)
                elif 'V2.' in self.model_type:
                    model = vggbuilder.build_vgg16_V2_variant(input_shape=self.size, weight_decay=self.weight_decay, \
                                                             num_classes=self.num_classes, config_code=self.variant)
        elif self.model_type == 'ResNet':
            model = resnetbuilder.build_resnet_18(input_shape=self.size, num_outputs=self.num_classes)
        elif self.model_type == 'Inception':
            base_model = InceptionV3(include_top=False, input_shape=self.size, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dropout(0.4)(x)
            predictions = Dense(self.num_classes, activation='sigmoid', name='fc')(x)
            model = Model(inputs=base_model.input, outputs=predictions)

            for layer in model.layers:
                if type(layer) == keras.layers.convolutional.Conv2D or type(layer) == keras.layers.Dense:
                    layer.kernel_regularizer = regularizers.l2(self.weight_decay)
        return model