
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def weight_variable(shape):
	initial = tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='VALID')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

def convolution_layer(x,filter_shape,pooling_shape):
	
	W=weight_variable(filter_shape)
	bias=[filter_shape[-1]]
	b=bias_variable(bias)
	conv=tf.nn.relu(conv2d(x,W)+b)
	y=tf.nn.max_pool(conv,ksize=pooling_shape,strides=pooling_shape,padding='VALID')
	return y
def fc_layer(x, in_node,out_node,keep_prob):
	W=weight_variable([in_node,out_node])
	b=bias_variable([out_node])
	x_reshape=tf.reshape(x,[-1,in_node])
	y_fc1=tf.nn.relu(tf.matmul(x_reshape,W)+b)
	y_fc=tf.nn.dropout(y_fc1,keep_prob=keep_prob)
	return y_fc
def fn_layer(x, in_node,out_node):
	W=weight_variable([in_node,out_node])
	b=bias_variable([out_node])
	x_reshape=tf.reshape(x,[-1,in_node])
	y_fn=tf.nn.relu(tf.matmul(x_reshape,W)+b)
	return y_fn
def convnn(x,keep_prob):
	x_image=tf.reshape(x,[-1,28,28,1])
	conv1=convolution_layer(x_image,filter_shape=[5,5,1,20],pooling_shape=[1,2,2,1])# 12*12*20
	conv2=convolution_layer(conv1,filter_shape=[5,5,20,40],pooling_shape=[1,2,2,1])# 4*4*40
	fc1=fc_layer(conv2,in_node=4*4*40,out_node=1000,keep_prob=keep_prob)
	fc2=fc_layer(fc1,in_node=1000,out_node=1000,keep_prob=keep_prob)
	net=fn_layer(fc2,in_node=1000,out_node=10)


	return net

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
x=tf.placeholder(tf.float32)
keep_prob=tf.placeholder(tf.float32)
y_=tf.placeholder(tf.float32)
y_conv=convnn(x,keep_prob)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for i in range(20000):
		batch=mnist.train.next_batch(50)
		if i %100 ==0:
			train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
			print 'step %d, training accuracy %g'%(i,train_accuracy)
		train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
	print 'test accuracy %g' % accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0})