import numpy as np 
def train(models,x_train,params):
    generator, discriminator, adversarial = models

    batch_size,latent_size,train_steps,model_name = params 

    save_interval = 500 #500�ܰ� ���� ������ �̹��� ���� 

    noise_input = np.random.uniform(-1,1,size=[16,latent_size]) #�Ʒ� ���� ������ ��ȭ Ȯ�� 

    train_size = x_train.shape[0]

    for i in range(train_steps):
        #��¥ ������ 
        rand_indexes = np.random.randint(0,train_size,size=batch_size)
        real_iamges = x_train[rand_indexes]

        #��¥ ������ 
        noise = np.random.uniform(-1,1,size=[batch_size,latent_size])
        fake_images = generator.predict(noise)

        #��¥ ������ + ��¥ ������ = �Ʒ� �������� 1 ��ġ 

        x = np.concatenate((real_iamges,fake_images))

        y = np.ones([2*batch_size,1]) # ���� ���̺� 
        y[batch_size:,:] = 0 #��¥ ���� ���̺� 

        #�Ǻ���, ��Ʈ��ũ �Ʒ�, �ս� ��Ȯ�� ��� 
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
                        
