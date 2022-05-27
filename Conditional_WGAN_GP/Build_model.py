from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Input, Reshape
from tensorflow.keras.layers import LeakyReLU, Activation,concatenate
from tensorflow.keras.layers import Cropping2D, ZeroPadding2D, UpSampling2D,Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as k 

class Discriminator():
    def __init__(self,inputs,half_condition,image_size=32):
        self.inputs = inputs
        self.condition = half_condition 
        self.image_size = image_size 
        self.kernel_size = 5 
        self.layer_filters = [32,64,128,256]

    def Conv_layer(self,filters,strides,x):
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(filters=filters,
                    kernel_size = self.kernel_size,
                    strides = strides,
                    padding = 'same')(x)
        return x 

    
    def build_dicriminator(self):
        #condition 
        y = Dense(2)(self.condition) #32*32*1
        y = Reshape((self.image_size,self.image_size,1))(y) #32,32,1
        #image data 
        x = self.inputs #32,32,1
        x = concatenate([x,y]) #32,32,2

        for filters in self.layer_filters:
            if filters == self.layer_filters[-1]:
                strides = 1 
            else:
                strides = 2 
            x = self.Conv_layer(filters,strides,x)
        
        x = Flatten()(x)
        x = Dense(1,activation = 'sigmoid')(x)
        
        discriminator = Model([self.inputs,self.condition],x,name = 'discriminator')
        return discriminator


class Generator():
    def __init__(self,noise_inputs,condition,image_size):
        self.inputs = noise_inputs 
        self.condition = condition 
        self.image_size = image_size 
        self.image_resize = self.image_size//4
        self.kernel_size = 5 
        self.condition_layers = [64,128,256,512]
        self.layer_filters = [128,64,32,1]

    def conv_T_layer(self,filters,strides,inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        outputs = Conv2DTranspose(filters = filters,
                            kernel_size = self.kernel_size,
                            strides = strides,
                            padding = 'same')(x)
        return outputs
    
    def conv_block(self,filters,use_bn,use_dropout,x):
        x = Conv2D(filters=filters,strides=2,padding='same',kernel_size=5,use_bias=True)(x)
        if use_bn:
            x = BatchNormalization()(x)
            x = LeakyReLU(0.2)(x)
        if use_dropout:
            x = Dropout(0.3)(x)
        return x


    def build_generator(self):
        """
        condition
        """
        x = self.condition 
        for n,filters in enumerate(self.condition_layers):
            if n in [1,2]:
                x = Conv2D(filters,strides=2,padding='same',kernel_size=5)(x)
                x = LeakyReLU(0.2)(x)
                x = Dropout(0.2)(x)
            else:
                x = Conv2D(filters,strides=2,padding='same',kernel_size=5)(x)
                x = LeakyReLU(0.2)(x)
        x = Flatten()(x)
        condition_out = x 

        """
        concat
        """
        x = concatenate([self.inputs,condition_out],axis=1)
        x = Dense(self.image_resize*self.image_resize*self.layer_filters[0])(x)
        x = Reshape((self.image_resize,self.image_resize,self.layer_filters[0]))(x)

        for filters in self.layer_filters:
            if filters > self.layer_filters[-2]:
                strides =2
            else:
                strides = 1 
            x = self.conv_T_layer(filters,strides,x)
        outputs = Activation('sigmoid')(x)
        generator = Model([self.inputs,self.condition],outputs,name = 'generator')
        return generator

class Adversarial():
    def __init__(self,generator,discriminator,inputs,condition):
        self.generator = generator
        self.discriminator = discriminator
        self.inputs = [inputs,condition]
        self.outputs = self.discriminator([self.generator(self.inputs),condition])
        
    
    def wasserstein_loss(y_label,y_pred):
        return -k.mean(y_label*y_pred) 
    
    def build_adversarial(self):
        self.adversarial = Model(self.inputs,self.outputs,name = 'CWGAN_gp')
        return self.adversarial