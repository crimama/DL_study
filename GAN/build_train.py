import numpy as np 
def train(models,x_train,params):
    generator, discriminator, adversarial = models

    batch_size,latent_size,train_steps,model_name = params 

    save_interval = 500 #500단계 마다 생성기 이미지 저장 

    noise_input = np.random.uniform(-1,1,size=[16,latent_size]) #훈련 동안 생성기 변화 확인 

    train_size = x_train.shape[0]

    for i in range(train_steps):
        #진짜 데이터 
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_iamges = x_train[rand_indexes]

        #가짜 데이터 
        noise = np.random.uniform(-1,1,size=[batch_size,latent_size])
        fake_images = generator.predict(noise)

        #진짜 데이터 + 가짜 데이터 = 훈련 데이터의 1 배치 

        x = np.concatenate((real_iamges,fake_images))

        y = np.ones([2*batch_size,1]) # 정답 레이블 
        y[batch_size:,:] = 0 #가짜 정답 레이블 

        #판별기, 네트워크 훈련, 손실 정확도 기록 
        loss,acc = discriminator.train_on_batch(x,y)
        log = "%d: [discriminator loss : %f, acc : %f]" % (i,loss,acc)


        noise = np.random.uniform(-1,1,size=[batch_size,latent_size])
        y = np.ones([batch_size,1])

        loss,acc = adversarial.train_on_batch(noise,y)
        log = "%s [adversarial loss :%f, acc :%f" % (log,loss,acc)

        print(log)

        if (i+1) % save_interval ==0:
            if (i+1) == train_steps:
                show = True
            else:
                show = False
            
            # plot_images(generator,
            #             noise_input=noise_input,
            #             show = show, 
            #             step  = (i+1),
            #             model_name = model_name)
                        
