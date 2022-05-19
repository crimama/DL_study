import numpy as np 
from tensorflow.keras.datasets import mnist,cifar10
from tensorflow import keras
import cv2 
def Mnist_load():
    (x_train,y_train), (_,_) = cifar10.load_data() #mnist 데이터 로드 

    #mnist 데이터 전처리 28,28 -> 28,28,1 / 정규화 
    temp_x_train = np.zeros((x_train.shape[0]*x_train.shape[1]*x_train.shape[2])).reshape(-1,x_train.shape[1],x_train.shape[2])
    for i in range(len(x_train)):
        temp_x_train[i] = cv2.cvtColor(x_train[i], cv2.COLOR_BGR2GRAY)
    x_train = temp_x_train

    image_size = x_train.shape[1] 
    x_train = x_train.reshape(-1,image_size,image_size,1)
    x_train = x_train.astype('float32')/255.

    num_labels = np.amax(y_train)+1 #라벨 갯수 : 0 ~ 9 라서 10개, 클래스 종류 
    y_labels = keras.utils.to_categorical(y_train)

    #파라미터 세팅 
    model_name = "cgan_mnist"
    latent_size = 100 
    batch_size =64 
    train_steps = 40000
    lr = 2e-4
    decay = 6e-8 
    input_shape = (image_size,image_size,1) #28,28,1
    label_shape = (num_labels,) #10

    data = (x_train, y_labels)
    params = (batch_size, latent_size, train_steps, num_labels, model_name)
    Shape = (input_shape,image_size,label_shape)
    learning = (lr,decay)

    return data, params,Shape,learning
