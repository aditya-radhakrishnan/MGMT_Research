import numpy as np
import tensorflow as tf
import math
from Preprocessor_Wrapper import PreprocessorWrapper

files_list = ['/home/raditya/Documents/untitled folder/multi1.txt',
                 '/home/raditya/Documents/untitled folder/multi2.txt',
                 '/home/raditya/Documents/untitled folder/multi3.txt']
models_list = ['/home/raditya/Documents/untitled folder/multimodel1.txt',
                 '/home/raditya/Documents/untitled folder/multimodel2.txt',
                 '/home/raditya/Documents/untitled folder/multimodel3.txt']
ratios_list = [0.5, 0.25, 0.25]
processor = PreprocessorWrapper(files_list, models_list, ratios_list, need_to_make_models=True)
preprocessor = processor.get_first_preprocessor()
'''
Three independent stacks
Two convolutional layers on each stack (max pooling and normalization after each layer)
Fully connected layer combines three stacks, followed by dropout
One convolutional layer
Fully connected layer, followed by dropout
Softmax layer
'''

# Count of training, test and validation sets
num_steps = 10000
# Document dimensions
dropout_keep_prob = 0.5
word_length = preprocessor.gensim_maker_obj.get_dimension_of_a_word()
num_words = preprocessor.max_num_words

train_s1 = processor.get_training_data_from_channel(channel_num=0)
train_s2 = processor.get_training_data_from_channel(channel_num=1)
train_s3 = processor.get_training_data_from_channel(channel_num=2)
train_labels = processor.get_training_data_labels()

test_s1 = processor.get_test_data_from_channel(channel_num=0)
test_s2 = processor.get_test_data_from_channel(channel_num=1)
test_s3 = processor.get_test_data_from_channel(channel_num=2)
test_labels = processor.get_test_data_labels()

valid_s1 = processor.get_validation_data_from_channel(channel_num=0)
valid_s2 = processor.get_validation_data_from_channel(channel_num=1)
valid_s3 = processor.get_validation_data_from_channel(channel_num=2)
valid_labels = processor.get_validation_data_labels()

num_train_docs = train_labels.shape[0]
num_test_docs = test_labels.shape[0]
num_valid_docs = valid_labels.shape[0]
num_labels = train_labels.shape[1]

# Neural network constants
batch_size = int(math.floor(num_train_docs / 50))

patch_size_s1_c1 = 20
patch_size_s1_c2 = 5
patch_size_s2_c1 = 20
patch_size_s2_c2 = 5
patch_size_s3_c1 = 20
patch_size_s3_c2 = 5

pool_size_s1_c1 = 2
pool_size_s1_c2 = 2
pool_size_s2_c1 = 2
pool_size_s2_c2 = 2
pool_size_s3_c1 = 2
pool_size_s3_c2 = 2

num_feat_s1_c1 = 16
num_feat_s1_c2 = 16
num_feat_s2_c1 = 16
num_feat_s2_c2 = 16
num_feat_s3_c1 = 16
num_feat_s3_c2 = 16

num_full_conn_1 = 32

patch_size_integ = 4
pool_size_integ = 2
num_feat_integ = 8
num_full_conn_2 = 32
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


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

graph = tf.Graph()


with graph.as_default():
    def gen_tf_data_labels(data_stack1, data_stack2, data_stack3, labels):
        return tf.constant(data_stack1, dtype=tf.float32), tf.constant(data_stack2, dtype=tf.float32), \
               tf.constant(data_stack3, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)

    tf_train_s1 = tf.placeholder(dtype=tf.float32)
    tf_train_s2 = tf.placeholder(dtype=tf.float32)
    tf_train_s3 = tf.placeholder(dtype=tf.float32)
    tf_train_labels = tf.placeholder(dtype=tf.float32)

    tf_test_s1, tf_test_s2, tf_test_s3, tf_test_labels = gen_tf_data_labels(test_s1, test_s2, test_s3, test_labels)
    tf_valid_s1, tf_valid_s2, tf_valid_s3, tf_valid_labels = gen_tf_data_labels(valid_s1, valid_s2, valid_s3,
                                                                                valid_labels)

    '''
    Setting up first two convolutional layers for stack 1
    '''

    def stack_model(tf_stack_data, patch_size_c1, patch_size_c2, num_feat_c1, num_feat_c2, pool_size_c1, pool_size_c2):
        print('tf_stack_data shape', tf_stack_data.get_shape())
        weights_c1, biases_c1 = weight_bias_variables([word_length, patch_size_c1, 1, num_feat_c1])
        weights_c2, biases_c2 = weight_bias_variables([1, patch_size_c2, num_feat_c1, num_feat_c2])

        first_conv_dim = int(math.floor((num_words - (patch_size_c1 - 1)) / pool_size_c1))
        second_conv_dim = int(math.floor((first_conv_dim - (patch_size_c2 - 1)) / pool_size_c2))
        size = second_conv_dim * 1 * num_feat_c2
        weights_full_conn1, biases_full_conn1 = weight_bias_variables([size, num_full_conn_1])

        h_conv1 = tf.nn.relu(conv2d(tf_stack_data, weights_c1) + biases_c1)
        h_pool1 = max_pool(h_conv1, pool_size_c1)
        h_norm1 = tf.nn.local_response_normalization(h_pool1)
        print('h_norm1 shape', h_norm1.get_shape())
        h_conv2 = tf.nn.relu(conv2d(h_norm1, weights_c2) + biases_c2)
        h_pool2 = max_pool(h_conv2, pool_size_c2)
        h_norm2 = tf.nn.local_response_normalization(h_pool2)
        print('h_norm2 shape', h_norm2.get_shape())
        h_norm2_flat = tf.reshape(h_norm2, [-1, size])
        return tf.matmul(h_norm2_flat, weights_full_conn1) + biases_full_conn1

    flat_integ_dim = int(math.floor((3 * num_full_conn_1 - (patch_size_integ - 1)) / pool_size_integ))
    reshaped_size = flat_integ_dim * num_feat_integ
    weights_conv_integ, biases_conv_integ = weight_bias_variables([1, patch_size_integ, 1, num_feat_integ])
    weights_full_conn2, biases_full_conn2 = weight_bias_variables([reshaped_size, num_full_conn_2])
    weights_output, biases_output = weight_bias_variables([num_full_conn_2, num_labels])

    def model(tf_s1, tf_s2, tf_s3, keep_prob=tf.constant(1.0)):
        h_s1 = stack_model(tf_s1, patch_size_s1_c1, patch_size_s1_c2, num_feat_s1_c1, num_feat_s1_c2,
                             pool_size_s1_c1, pool_size_s1_c2)
        h_s2 = stack_model(tf_s2, patch_size_s2_c1, patch_size_s2_c2, num_feat_s2_c1, num_feat_s2_c2,
                             pool_size_s2_c1, pool_size_s2_c2)
        h_s3 = stack_model(tf_s3, patch_size_s3_c1, patch_size_s3_c2, num_feat_s3_c1, num_feat_s3_c2,
                             pool_size_s3_c1, pool_size_s3_c2)
        # print('h_s1 shape', h_s1.get_shape())
        h_full_conn_1 = tf.nn.relu(tf.concat(1, [h_s1, h_s2, h_s3]))
        h_drop1 = tf.nn.dropout(h_full_conn_1, keep_prob)
        # print('h_drop1', h_drop1.get_shape())
        h_reshaped = tf.reshape(h_drop1, [-1, 1, 3 * num_full_conn_1, 1])
        # print('h_reshaped', h_reshaped.get_shape())
        h_conv_integ = tf.nn.relu(conv2d(h_reshaped, weights_conv_integ) + biases_conv_integ)
        h_pool_integ = max_pool(h_conv_integ, pool_size_integ)
        h_norm_integ = tf.nn.local_response_normalization(h_pool_integ)
        # print('h_norm_integ', h_norm_integ.get_shape())
        h_flat_integ = tf.reshape(h_norm_integ, [-1, reshaped_size])
        # print('h_flat_integ shape', h_flat_integ.get_shape())
        h_full_conn2 = tf.nn.relu(tf.matmul(h_flat_integ, weights_full_conn2) + biases_full_conn2)
        h_drop2 = tf.nn.dropout(h_full_conn2, keep_prob)
        # print('h_drop2 shape', h_drop2.get_shape())
        return tf.matmul(h_drop2, weights_output) + biases_output

    logits = model(tf_train_s1, tf_train_s2, tf_train_s3, dropout_keep_prob)

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(weights_output))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_s1, tf_test_s2, tf_test_s3))
    valid_prediction = tf.nn.softmax(model(tf_valid_s1, tf_valid_s2, tf_valid_s3, dropout_keep_prob))

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        batch_indices = np.random.choice(num_train_docs, size=batch_size, replace=False)
        batch_s1 = train_s1[batch_indices, :, :, :]
        batch_s2 = train_s2[batch_indices, :, :, :]
        batch_s3 = train_s3[batch_indices, :, :, :]
        batch_labels = train_labels[batch_indices, :]
        feed_dict = {tf_train_s1: batch_s1, tf_train_s2: batch_s2, tf_train_s3: batch_s3, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if step % 500 == 0:
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))