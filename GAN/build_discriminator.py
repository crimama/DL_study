from tensorflow.keras.layers import Dense, Reshape,BatchNormalization,Activation,Conv2DTranspose,LeakyReLU,Conv2D,Flatten,Input
from tensorflow.keras.models import Model

def build_discriminator(inputs):
    kernel_size =5
    layer_filters = [32,64,128,256]

    x = inputs
    for filters in layer_filters:
        if filters == layer_filters[-1]:
            strides = 1 
        else:
            strides = 2 
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters = filters,
                    kernel_size = kernel_size,
                    strides = strides,
                    padding = 'same')(x)
    x = Flatten()(x)
    x = Dense(1)(x)
    x = Activation('sigmoid')(x)
    discriminator = Model(inputs,x,name = 'discriminator')
    return discriminator