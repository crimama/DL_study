from tensorflow.keras.layers import Dense,Reshape,BatchNormalization,Activation,Conv2DTranspose
from tensorflow.keras.models import Model 


def build_generator(inputs, image_size):
    #积己扁 葛胆 备己
    image_resize = image_size//4
    kernel_size = 5  
    layer_filters = [128,64,32,1]

    x = Dense(image_resize*image_resize*layer_filters[0])(inputs)
    x = Reshape((image_resize,image_resize,layer_filters[0]))(x)

    for filters in layer_filters:
        if filters > layer_filters[-2]:
            strides = 2 
        else:
            strides = 1 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters = filters,
                            kernel_size = kernel_size,
                            strides = strides,
                            padding = 'same')(x)
    x = Activation('sigmoid')(x)
    generator = Model(inputs,x, name = 'generator')
    return generator 