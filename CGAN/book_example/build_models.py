from tensorflow.keras.layers import Dense, Reshape,BatchNormalization,Activation,Conv2DTranspose,LeakyReLU,Conv2D,Flatten,Input,concatenate
from tensorflow.keras.models import Model

class Discriminator():
    def __init__(self,inputs,y_labels,image_size):
        self.inputs = inputs
        self.y_labels = y_labels 
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
        y = Dense(self.image_size*self.image_size)(self.y_labels)
        y = Reshape((self.image_size,self.image_size,1))(y)
        x = self.inputs
        x = concatenate([x,y])

        for filters in self.layer_filters:
            if filters == self.layer_filters[-1]:
                strides = 1 
            else:
                strides = 2 
            x = self.Conv_layer(filters,strides,x)
        
        x = Flatten()(x)
        x = Dense(1,activation = 'sigmoid')(x)
        
        discriminator = Model([self.inputs,self.y_labels],x,name = 'discriminator')
        return discriminator

class Adversarial():
    def __init__(self,generator,discriminator,inputs,labels,optimizer):
        self.generator = generator
        self.discriminator = discriminator
        self.inputs = [inputs,labels]
        self.outputs = self.discriminator([self.generator(self.inputs),labels])
        self.optimizer = optimizer
        
    
    def build_adversarial(self):
        self.adversarial = Model(self.inputs,self.outputs,name = 'cgan_mnist')
        self.adversarial.compile(loss = 'binary_crossentropy',
                    optimizer = self.optimizer,
                    metrics = ['accuracy'])
        
        
        return self.adversarial

class Generator():
    def __init__(self,inputs,y_labels,image_size):
        self.inputs = inputs 
        self.y_labels = y_labels 
        self.image_size = image_size 
        self.image_resize = self.image_size//4
        self.kernel_size = 5 
        self.layer_filters = [128,64,32,1]

    def conv_T_layer(self,filters,strides,inputs):
        x = BatchNormalization()(inputs)
        x = Activation('relu')(x)
        outputs = Conv2DTranspose(filters = filters,
                            kernel_size = self.kernel_size,
                            strides = strides,
                            padding = 'same')(x)
        return outputs

    def build_generator(self):
        x = concatenate([self.inputs,self.y_labels],axis=1)
        x = Dense(self.image_resize*self.image_resize*self.layer_filters[0])(x)
        x = Reshape((self.image_resize,self.image_resize,self.layer_filters[0]))(x)

        for filters in self.layer_filters:
            if filters > self.layer_filters[-2]:
                strides =2
            else:
                strides = 1 
            x = self.conv_T_layer(filters,strides,x)
        outputs = Activation('sigmoid')(x)
        generator = Model([self.inputs,self.y_labels],outputs,name = 'generator')
        return generator