import numpy as np 
import matplotlib.pyplot as plt
import os 
import math 

class CGAN():
    def __init__(self,models,data,params):
        (self.x_train, self.y_train)  = data 
        (self.batch_size, self.latent_size, self.train_steps, self.num_labels, self.model_name) = params
        (self.adversarial,self.generator,self.discriminator) = models
        #�н� �߰� Ȯ�� �� valid ������ 
        self.valid_noise_input = np.random.uniform(-1.0,1.0,size=[16,self.latent_size]) #shape = (16,100)
        self.valid_condition = np.eye(self.num_labels)[np.arange(0,16) % self.num_labels] #�� �� ���� ����
        #�н� �� ������ �� 
        self.train_size = self.x_train.shape[0] #60000
        self.save_interval =500
        print(self.model_name,"labels for generated images:",np.argmax(self.valid_condition,axis=1)) #�������ڵ� -> ī�װ��� ���ڵ� 

    def plot_images(self):
        images = self.generator.predict([self.valid_noise_input,self.valid_condition])
        num_images = images.shape[0] #16
        image_size = images.shape[1] #28

        plt.figure(figsize=(2.2, 2.2))
        rows = int(math.sqrt(self.valid_noise_input.shape[0]))
        for i in range(num_images):
            image = np.reshape(images[i], [image_size, image_size])
            plt.subplot(rows, rows, i + 1)
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.show()

    def create_data(self):
        #��¥ ������ ���� 
        rand_indexes = np.random.randint(0,self.train_size,size=self.batch_size)#6���� �� 16�� ���� 
        real_images = self.x_train[rand_indexes]
        real_labels = self.y_train[rand_indexes] #��¥ ���̺� (����)

        #��¥ ������ ���� 
        noise = np.random.uniform(-1.0,1.0, size=[self.batch_size,self.latent_size]) #(64,100)���·� ������ ����
        fake_labels = np.eye(self.num_labels)[np.random.choice(self.num_labels,self.batch_size)] #��¥ ���̺�(����)
        fake_images = self.generator.predict([noise,fake_labels]) #���� + ������ +������ -> ��¥ �̹���

        #�Ʒ� ������ ���� = ��¥ ������ + ��¥ ������ 
        x = np.concatenate((real_images, fake_images)) #shape = (128,28,28,1)
        condition = np.concatenate((real_labels,fake_labels)) #shape = (128,10) �������ڵ��� 

        y = np.ones([2*self.batch_size,1])
        y[self.batch_size:,:] = 0 #ī�װ��� �󺧸� shape = (128,1)
        
        return x,y,condition
        
    def train(self):
        for i in range(self.train_steps):

            """
            discriminator metric
            """
            #�򰡿� ������ �ε� 
            x,y,condition = self.create_data()

            #�Ǻ��� ��Ʈ��ũ �Ʒ�, �սǰ� ��Ȯ�� ��� 
            loss,acc = self.discriminator.train_on_batch([x,condition],y)
            log = "%d : [discriminator loss : %f, acc: %f]" % (i,loss,acc)

            """
            adversarial 
            """
            # ������ input ���� 
            noise = np.random.uniform(-1.0,1.0,size = [self.batch_size,self.latent_size]) #noise shape = 64,100
            condition = np.eye(self.num_labels)[np.random.choice(self.num_labels,self.batch_size)] #condition 64,10

            # ������ output ���� 
            y = np.ones([self.batch_size,1]) #shape = 64,1

            # ������ �Ű�� �н� �� �� 
            loss,acc = self.adversarial.train_on_batch([noise,condition],y)
            log = "%s [adversarial loss :%f, acc :%f" % (log,loss,acc)
            
            if (i+1) % self.save_interval == 0:
                print(log)
                self.plot_images()
                            
        self.generator.save(f'{self.model_name}.h5')