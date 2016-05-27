import numpy as np
import tensorflow as tf
import math
from Preprocessing import Preprocessing

processor = Preprocessing('/home/raditya/Documents/untitled folder/part1.txt',
                          '/home/raditya/Documents/untitled folder/EarningsCallModelPart1.txt', need_to_create_model=True)

# Count of training, test and validation sets
num_steps = 1
# Document dimensions
dropout_keep_prob = 0.5
word_length = processor.gensim_maker_obj.get_dimension_of_a_word()
num_words = processor.max_num_words

train_data, train_values, valid_data, valid_values, test_data, test_values = processor.get_data_set([0.5, 0.25, 0.25])
num_train_docs = train_values.shape[0]
num_test_docs = test_values.shape[0]
num_valid_docs = valid_values.shape[0]
num_output_values = train_values.shape[1]

# Neural network constants
batch_size = int(math.floor(num_train_docs / 10))
patch_sizes1 = 10
patch_sizes2 = 5
pooling_size = 4
num_features_conv1 = 16
num_features_conv2 = 16
num_full_conn = 32
beta = 0.01


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)


def weight_bias_variables(shape):
    return weight_variable(shape), bias_variable([shape[len(shape) - 1]])

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, num_filter_words):
  return tf.nn.max_pool(x, ksize=[1, 1, num_filter_words, 1], strides=[1, 1, num_filter_words, 1], padding='VALID')


def accuracy(predicts, values):
    return predicts - values
  # return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])
graph = tf.Graph()


with graph.as_default():
    def gen_tf_data_values(data, values):
        return tf.constant(data, dtype=tf.float32), tf.constant(values, dtype=tf.float32)

    tf_train_data = tf.placeholder(dtype=tf.float32)
    tf_train_values = tf.placeholder(dtype=tf.float32)
    tf_test_data, tf_test_values = gen_tf_data_values(test_data, test_values)
    tf_valid_data, tf_valid_values = gen_tf_data_values(valid_data, valid_values)

    # Weights are of the shape [vector_length, patch_size, num_features], biases are of the shape [num_features]
    weights_c1, biases_c1 = weight_bias_variables([word_length, patch_sizes1, 1, num_features_conv1])

    # Second set of weights
    weights_c2, biases_c2 = weight_bias_variables([1, patch_sizes2, num_features_conv1, num_features_conv2])

    # Weights and biases for full connected layer
    first_conv_dim = int(math.floor((num_words - (patch_sizes1 - 1)) / pooling_size))
    second_conv_dim = int(math.floor((first_conv_dim - (patch_sizes2 - 1)) / pooling_size))
    size = second_conv_dim * 1 * num_features_conv2
    weights_full_conn, biases_full_conn = weight_bias_variables()([size, num_full_conn])

    # Weights and biases for softmax layer
    weights_output, biases_output = weight_bias_variables([num_full_conn, num_output_values])

    def model(tf_data, keep_prob=tf.constant(1.0)):
        h_conv1 = tf.nn.tanh(conv2d(tf_data, weights_c1) + biases_c1)
        # print('h_conv1 shape', h_conv1.get_shape())
        h_pool1 = max_pool(h_conv1, pooling_size)
        # print('h_pool1 shape', h_pool1.get_shape())
        h_norm1 = tf.nn.local_response_normalization(h_pool1)
        h_conv2 = tf.nn.tanh(conv2d(h_norm1, weights_c2) + biases_c2)
        # print('h_conv2 shape', h_conv2.get_shape())
        h_pool2 = max_pool(h_conv2, pooling_size)
        # print('h_pool2 shape', h_pool2.get_shape())
        h_norm2 = tf.nn.local_response_normalization(h_pool2)
        h_norm2_flat = tf.reshape(h_norm2, [-1, size])
        # print('h_norm2_flat shape', h_norm2_flat.get_shape())
        h_full_conn = tf.nn.tanh(tf.matmul(h_norm2_flat, weights_full_conn) + biases_full_conn)
        # print('h_full_conn shape', h_full_conn.get_shape())
        h_drop = tf.nn.dropout(h_full_conn, keep_prob)
        # print('h_drop shape', h_drop.get_shape())
        return tf.matmul(h_drop, weights_output) + biases_output

    train_prediction = model(tf_train_data, dropout_keep_prob)

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.square(train_prediction - tf_train_values))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    test_prediction = model(tf_test_data)
    valid_prediction = model(tf_valid_data)

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        batch_indices = np.random.choice(num_train_docs, size=batch_size, replace=False)
        batch_data = train_data[batch_indices, :, :, :]
        batch_values = train_values[batch_indices, :]
        feed_dict = {tf_train_data: batch_data, tf_train_values: batch_values}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_values))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_values))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_values))