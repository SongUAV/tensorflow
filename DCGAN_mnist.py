import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data



from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout,LeakyReLU,Activation
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
from matplotlib import pyplot as plt
import numpy as np


import matplotlib.pyplot as plt 


def discriminator():
	D=Sequential()
	depth=32
	dropout=0.4
	input_shape=(1,28,28)
	# in 28*28*1
	# out 7*7*256
	# 14*14*32
	D.add(Conv2D(depth*1,3,strides=2,padding='same',activation=LeakyReLU(alpha=0.2),input_shape=input_shape))
	D.add(Dropout(dropout))
	# 14*14*64
	D.add(Conv2D(depth*2,3,strides=1,padding='same',activation=LeakyReLU(alpha=0.2)))
	D.add(Dropout(dropout))
	# 7*7*128
	D.add(Conv2D(depth*4,3,strides=2,padding='same',activation=LeakyReLU(alpha=0.2)))
	D.add(Dropout(dropout))
	# 7*7*256
	D.add(Conv2D(depth*8,3,strides=1,padding='same',activation=LeakyReLU(alpha=0.2)))
	D.add(Dropout(dropout))
	# flatten
	D.add(Flatten())

	image=Input(shape=input_shape)

	features=D(image)

	prob=Dense(1,activation='sigmoid',name='probibility')(features)
	#digit=Dense(10,activation='softmax',name='digit')(features)

	return Model(input=image,output=prob)

def generator(latent_size):
	dropout = 0.4
	depth = 64+64+64+64
	dim = 7 

	G=Sequential()
	G.add(Dense(dim*dim*depth,input_dim=latent_size))
	G.add(Activation('relu'))
	# 7,7,256
	G.add(Reshape((depth,dim,dim)))
	G.add(Dropout(dropout))

	# 14,14,128
	G.add(UpSampling2D())
	G.add(Conv2DTranspose(int(depth/2),3,padding='same'))
	G.add(Activation('relu'))

	# 28,28,64
	G.add(UpSampling2D())
	G.add(Conv2DTranspose(int(depth/4),3,padding='same'))
	G.add(Activation('relu'))	

	# 28,28,32
	G.add(Conv2DTranspose(int(depth/8),3,padding='same'))
	G.add(Activation('relu'))

	# 28,28,1
	G.add(Conv2DTranspose(1,3,padding='same'))
	G.add(Activation('tanh'))

	latent=Input(shape=(latent_size,))

	#image_digit=Input(shape=(1,),dtype='int32')

	#digits=Flatten()(Embedding(10,latent_size,init='glorot_normal')(image_digit))

	#digit_embedding=merge([latent,digits],mode='mul')

	#fake_image=G(digit_embedding)
	fake_image=G(latent)

	return Model(input=latent,output=fake_image)



nb_epochs=100
batch_size=100
latent_size=100

adam_lr=0.0002
adma_beta_1=0.5

D=discriminator()
#D.compile(optimizer=Adam(lr=adam_lr*5,beta_1=adma_beta_1),loss=['binary_crossentropy','sparse_categorical_crossentropy'])
D.compile(optimizer=Adam(lr=adam_lr*2,beta_1=adma_beta_1),loss='binary_crossentropy')

G=generator(latent_size)
#G.compile(optimizer=Adam(lr=adam_lr*1,beta_1=adma_beta_1),loss='binary_crossentropy')

latent=Input(shape=(latent_size,))
#image_class=Input(shape=(1,),dtype='int32')

fake_image=G(latent)

D.trainable=False
prob=D(fake_image)

AD=Model(input=latent,output=prob)
AD.compile(optimizer=Adam(lr=adam_lr*1,beta_1=adma_beta_1),loss='binary_crossentropy')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train =(x_train.astype(np.float32)-127.5)/127.5
x_train=np.expand_dims(x_train,axis=1)

x_test=(x_test.astype(np.float32)-127.5)/127.5
x_test=np.expand_dims(x_test,axis=1)
nb_train,nb_test=x_train.shape[0],x_test.shape[0]

noise_input=np.random.uniform(-1,1,(batch_size,latent_size))

for epoch in range(nb_epochs):
	print ('Epoch {} of {}'.format(epoch+1,nb_epochs))

	nb_batches=int(x_train.shape[0] / batch_size)
	progress_bar=Progbar(target=nb_batches)

	epoch_gen_loss =[]
	epoch_disc_loss=[]

	np.random.shuffle(x_train)
	for index in range(nb_batches):
		progress_bar.update(index)

		noise=np.random.uniform(-1,1,(batch_size,latent_size))
		

		image_batch=x_train[index * batch_size:(index+1)*batch_size]
		#label_batch=y_train[index * batch_size:(index+1)*batch_size]


		#sampled_labels =np.random.randint(0,10,batch_size)
		generated_images=G.predict(noise,verbose=0)

		x=np.concatenate((image_batch,generated_images))
		y=np.array([1]*batch_size+[0]*batch_size)
		#aux_y=np.concatenate((label_batch,sampled_labels),axis=0)

		epoch_disc_loss.append(D.train_on_batch(x,y))

		noise=np.random.uniform(-1,1,(2*batch_size,latent_size))
		#sampled_labels=np.random.randint(0,10,2*batch_size)


		y=np.ones(2*batch_size)

		epoch_gen_loss.append(AD.train_on_batch(noise,y))
	#print ('\nTesting for epoch{}:'.format(epoch+1))

	noise=np.random.uniform(-1,1,(nb_test,latent_size))
	#sampled_labels=np.random.randint(0,10,nb_test)
	generated_images=G.predict(noise,verbose=False)

	x=np.concatenate((x_test,generated_images))
	y=np.array([1]*nb_test+[0]*nb_test)
	#aux_y=np.concatenate((y_test,sampled_labels),axis=0)

	discriminator_test_loss=D.evaluate(x,y,verbose=False)
	discriminator_train_loss=np.mean(np.array(epoch_disc_loss),axis=0)

	noise=np.random.uniform(-1,1,(2*nb_test,latent_size))
	sampled_labels=np.random.randint(0,10,2*nb_test)

	y=np.ones(2*nb_test)
	generator_test_loss=AD.evaluate(noise,y,verbose=False)
	generator_train_loss=np.mean(np.array(epoch_gen_loss),axis=0)

	print ("\n generator_test_loss is %f " % generator_test_loss)
	print ("\n generator_train_loss is %f " % generator_train_loss)
	print ("\n discriminator_test_loss is %f " % discriminator_test_loss)
	print ("\n discriminator_train_loss is %f " % discriminator_train_loss)
	#noise=np.random.uniform(-1,1,(100,latent_size))
	#sampled_labels=np.array([[i]*10 for i in range(10)]).reshape(-1,1)
	generated_images=G.predict(noise_input,verbose=0)
	plt.figure(figsize=(10,10))
	for i in range(generated_images.shape[0]):
		plt.subplot(10,10,i+1)
		image=generated_images[i,:,:,:]
		image=np.reshape(image,[28,28])
		plt.imshow(image,cmap='gray')
		plt.axis('off')
	filename = "mnist_%d.png" % epoch
	plt.tight_layout()
	plt.savefig(filename)
	plt.close('all')
