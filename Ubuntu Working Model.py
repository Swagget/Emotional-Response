import os
from PIL import Image
'''from scipy import misc'''
import numpy as np
import tensorflow as tf


# from tensorflow.examples.tutorials.mnist import input_data

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def classify_the_images():
	global dict
	global emos
	dict = {}
	emos = 0
	files = os.listdir("/media/excelcior/Stuff/Programming/Deep learning/Emotional Response/Resized Photos")
	for x in files:
		strin = str(x)
		position = strin.find("_")
		emotion = strin[0:position]
		if emotion not in dict:
			dict[emotion] = emos
			emos += 1
	print(len(dict))

'''
def normalise_the_images():
	files = os.listdir("Z:\\Programming\\Deep learning\\Emotional Response\\testImages_artphoto")
	os.chdir("Z:\\Programming\\Deep learning\\Emotional Response\\Resized Photos")
	print("here")
	for x in files:
		img = Image.open("Z:\\Programming\\Deep learning\\Emotional Response\\testImages_artphoto\\" + str(x))
		img.load()
		# print("here1")
		image_arr = np.asarray(img)
		# print(img)
		# print("here2")
		resized_image = misc.imresize(image_arr, [600, 600])
		# print(resized_image)
		# print("here3")
		re_encoded = Image.fromarray(resized_image)
		# print(re_encoded)
		# print("here4")
		re_encoded.save("Z:\\Programming\\Deep learning\\Emotional Response\\Resized Photos\\" + str(x))
		print(x)

'''
classify_the_images()


# normalise_the_images()
def create_training_set_data_table():
	# take all the images
	# Convert them all into a numpy array format
	# store them in a table along with their output labels as a different column

	os.chdir("/media/excelcior/Stuff/Programming/Deep learning/Emotional Response/Resized Photos")
	files = os.listdir("/media/excelcior/Stuff/Programming/Deep learning/Emotional Response/Resized Photos")
	global dataset
	capacity = 1
	size = 0
	dataset = np.zeros((1080008))  # Size of  image as an array + the one-hot array of labels
	for x in files:
		img = Image.open("/media/excelcior/Stuff/Programming/Deep learning/Emotional Response/Resized Photos/" + str(x))
		img.load()
		image_arr = np.asarray(img)
		# print(image_arr.shape)
		# print(image_arr.size)
		# print(image_arr[0])
		# print(image_arr.shape)
		# print(x)
		if (image_arr.size == 360000):
			print("Triggered")
			# print(image_arr.dtype)
			temp_arr = np.zeros(360000 * 3, dtype=image_arr.dtype)
			temp_arr = np.reshape(temp_arr, (600, 600, 3))
			for t1 in range(600):
				for t2 in range(600):
					temp_arr[t1][t2][0] = image_arr[t1][t2]
					temp_arr[t1][t2][1] = image_arr[t1][t2]
					temp_arr[t1][t2][2] = image_arr[t1][t2]
			image_arr = temp_arr
		image_arr = np.reshape(image_arr, (1080000))
		# print(image_arr.shape)

		emotion = dict[x.split("_")[0]]
		emotion_array = np.zeros(len(dict))
		emotion_array[emotion] = 1
		# print(emotion_array)
		# print(emotion_array.shape)
		image_and_emotion_array = np.concatenate([image_arr, emotion_array], axis=0)
		# dataset = np.concatenate([dataset,image_and_emotion_array[:,None]], axis=1)
		if size == capacity:
			capacity *= 4
			newdata = np.zeros((capacity * 1080008,))
			# print("size is")
			# print(dataset.shape)
			newdata[:size * 1080008] = dataset
			dataset = newdata
		dataset[size * 1080008:(size + 1) * 1080008] = image_and_emotion_array
		size += 1
		print(size)
	dataset = dataset[:size * 1080008]
	dataset = np.reshape(dataset, (1080008, size))
	return dataset

'''
def X_next_batch(batch_size):
	print("got to the function X")
	batch_no += 1
	batch = tf.variable(batch_no)
	batch %= 13
	current_batch = batch
	array_of_images = []
	files = os.listdir("Z:\\Programming\\Deep learning\\Emotional Response\\Resized Photos")
	while (True):
		array_of_images.append(files[current_batch])
		current_batch += 13
		if (current_batch >= 806):
			break
	return array_of_images


def Y_next_batch(batch_size, number):
	print("got to the function Y")
	files = os.listdir("Z:\\Programming\\Deep learning\\Emotional Response\\Resized Photos")
	current_batch = number
	image_number = 0
	array_of_outputs = np.zeros(62, n_classes)
	while (True):
		strin = str(files[current_batch])
		position = strin.find("_")
		emotion = strin[0:position]
		output_number = dict[emotion]
		array_of_outputs[image_number][output_number] = 1
		current_batch += 13
		if (current_batch >= 806):
			break
		image_number += 1
	return array_of_outputs
'''

n_classes = tf.constant(emos)
batch_size = 62
batch = tf.constant(0)

x = tf.placeholder('float', [None, 1080000])
y = tf.placeholder('float')


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def convolutional_neural_network(data):
	weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, 3, 32])),
			   # 5*5 Convolution 1 input and 32]] features or outputs
			   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64])),
			   # 5*5 Convolution 32 inputs and 64 features or outputs
			   'W_fc'   : tf.Variable(tf.random_normal([7 * 7 * 64, 1024])),
			   'out'    : tf.Variable(tf.random_normal([1024, n_classes])),
			   }

	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			  'b_conv2': tf.Variable(tf.random_normal([64])),
			  'b_fc'   : tf.Variable(tf.random_normal([1024])),
			  'out'    : tf.Variable(tf.random_normal([n_classes])),
			  }
	print(data.shape)
	reshaped = tf.reshape(data, shape=[-1, 600, 600,3])  # The problem is that this here is the prototype and this alone is using more than 2 gb
	print("Got to the point model")
	conv1 = conv2d(reshaped, weights['W_conv1']) + biases['b_conv1']
	conv1 = maxpool2d(conv1)

	conv2 = conv2d(conv1, weights['W_conv2']) + biases['b_conv2']
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, (-1, 7 * 7 * 64))
	fc = tf.nn.relu(tf.matmul(fc, weights["W_fc"]) + biases['b_fc'])

	output = tf.matmul(fc, weights["out"]) + biases['out']

	return output


def train_neural_network(x, data):
	print("Running the main function")
	print(x.shape)
	prediction = convolutional_neural_network(x)
	# y = tf.placeholder('float')
	# OLD VERSION:
	# cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
	# NEW:
	print("Created the tensors")

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)
	print("created optimiser and cost functions")
	arr_X = x[:1080001][:]
	arr_Y = x[1080001:][:]
	hm_epochs = 10
	shuffeled_index_numbers = np.random.permutation(806)
	temp2 = 0
	'''
	X_randomed = data[0:1080000,0:1080000]
	Y_randomed = data[0:806,1080000:1080008]
	print("Shapes are")
	print(data.shape)
	print("data ")
	#print(arr_X.shape)
	print(X_randomed.shape)
	print(Y_randomed.shape)
	print("Done")

	for temp in shuffeled_index_numbers:
		for temp3 in range(1080000):
			print(x[temp][temp3])
			X_randomed[temp2][temp3] = x[temp][temp3]
			print(x[temp][temp3])
		for temp3 in range(1080000,1080008):
			Y_randomed[temp2][temp3-1080000] = x[temp][temp3]
		temp2+=1
	'''
	temp_data = data
	np.random.shuffle(temp_data)
	print("shuffled the array")

	X_randomed = temp_data[0:1080000, 0:1080000]
	Y_randomed = temp_data[0:806, 1080000:1080008]

	print("Split the shuffled array")

	with tf.Session() as sess:
		# OLD:
		# sess.run(tf.initialize_all_variables())
		# NEW:
		sess.run(tf.global_variables_initializer())
		print("Initilized all the variables")
		for epoch in range(hm_epochs):
			epoch_loss = 0
			for i in range(13):
				print("Even reached here")
				epoch_x = X_randomed[i * 62: (i + 1) * 62]
				epoch_y = Y_randomed[i * 62: (i + 1) * 62]
				print("About to run it")
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
				print("Completed the epoch", epoch, "And loss is ", epoch_loss)

			print('Epoch', epoch + 1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


dataset = np.transpose(create_training_set_data_table())
print("This much worked")
train_neural_network(x, dataset)
