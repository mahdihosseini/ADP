import keras
import vggbuilder
import resnetbuilder
import histonetbuilder
import xceptionbuilder
import mobilenetbuilder
from keras.applications import InceptionV3

from keras.layers import Dropout, GlobalAveragePooling2D, Dense
from keras import regularizers



class ModelLoader:
    # Initialize
    def __init__(self, params):
        self.model_type = params['model']
        self.variant = params['variant']
        self.num_classes = params['num_classes']
        self.img_size = params['img_size']
        self.size = [self.img_size, self.img_size, 3]
        self.weight_decay = 5e-4

    # Load and customize default Keras networks
    def load_model(self):
        if self.model_type == 'VGG':
            if self.variant == 'Default':
                model = vggbuilder.build_vgg16(input_shape=self.size, weight_decay=self.weight_decay, \
                                               num_classes=self.num_classes, num_blocks=5)
            else:
                False
        elif self.model_type == 'ResNet':
            if self.variant =='resnet_18':
                model = resnetbuilder.build_resnet_18(self.size, self.num_classes)
            elif self.variant == 'resnet_34':
                model = resnetbuilder.build_resnet_34(self.size, self.num_classes)
            elif self.variant == 'resnet_50':
                model = resnetbuilder.build_resnet_50(self.size, self.num_classes)
            else:
                False
        elif self.model_type == 'Xception':
            if self.variant == 'Xception_V1':
                model = xceptionbuilder.Xception(self.size, self.num_classes)
            else:
                False
        elif self.model_type == 'MobileNet':
            if self.variant == 'V1':
                model = mobilenetbuilder.MobileNet(self.size, self.num_classes)
            else:
                False
        elif self.model_type == 'Inception':
            base_model = InceptionV3(include_top=False, input_shape=self.size, weights=None)
            x = base_model.output
            x = GlobalAveragePooling2D(name='avg_pool')(x)
            x = Dropout(0.4)(x)
            predictions = Dense(self.num_classes, activation='sigmoid', name='fc')(x)
            model = Model(inputs=base_model.input, outputs=predictions)
        elif self.model_type == 'HistoNet':
            model = histonetbuilder.build_histonet_series_1(input_shape=self.size, weight_decay=self.weight_decay, num_classes=self.num_classes, config_code=self.variant)
        return model