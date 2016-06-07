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

ratios_list = [0.7, 0.15, 0.15]
processor = PreprocessorWrapper(files_list, models_list, ratios_list, need_to_make_models=False)
preprocessor = processor.get_first_preprocessor()
'''
Three independent stacks
One convolutional layers on each stack
Fully connected layer combines three stacks
Fully connected layer
Softmax layer
'''

# Count of training, test and validation sets
num_steps = 50000
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

# Constants for first stack
patch_size_s1_c1 = 5
pool_size_s1_c1 = 4
num_feat_s1_c1 = 16

patch_size_s1_c2 = 3
pool_size_s1_c2 = 4
num_feat_s1_c2 = 16

# Constants for second stack
patch_size_s2_c1 = 5
pool_size_s2_c1 = 4
num_feat_s2_c1 = 16

patch_size_s2_c2 = 3
pool_size_s2_c2 = 4
num_feat_s2_c2 = 16

# Constants for third stack
patch_size_s3_c1 = 5
pool_size_s3_c1 = 4
num_feat_s3_c1 = 16

patch_size_s3_c2 = 3
pool_size_s3_c2 = 4
num_feat_s3_c2 = 16

# Combined network constants
num_feat_fc1 = 16
keep_prob1 = 0.5

patch_size_comb_c1 = 2
pool_size_comb_c1 = 4
num_feat_comb_c1 = 16

num_feat_fc2 = 16
keep_prob2 = 0.5

batch_size = int(math.floor(num_train_docs / 100))
beta = 0.01


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def weight_bias_variables(shape):
    return weight_variable(shape), bias_variable([shape[len(shape) - 1]])


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, num_filter_words):
    return tf.nn.max_pool(x, ksize=[1, 1, num_filter_words, 1], strides=[1, 1, num_filter_words, 1], padding='VALID')


def accuracy(predicts, labels):
    return 100.0 * np.sum(np.argmax(predicts, 1) == np.argmax(labels, 1)) / predicts.shape[0]


def tensor_print(label, tensor):
    if tensor.get_shape()[0].value is not None:
        print(label + ' shape', tensor.get_shape())

graph = tf.Graph()


with graph.as_default():
    def gen_tf_data_labels(data_s1, data_s2, data_s3, labels):
        return tf.constant(data_s1, dtype=tf.float32), tf.constant(data_s2, dtype=tf.float32), \
            tf.constant(data_s3, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)

    tf_train_s1 = tf.placeholder(dtype=tf.float32)
    tf_train_s2 = tf.placeholder(dtype=tf.float32)
    tf_train_s3 = tf.placeholder(dtype=tf.float32)
    tf_train_labels = tf.placeholder(dtype=tf.float32)

    tf_test_s1, tf_test_s2, tf_test_s3, tf_test_labels = gen_tf_data_labels(test_s1, test_s2, test_s3, test_labels)
    tf_valid_s1, tf_valid_s2, tf_valid_s3, tf_valid_labels = gen_tf_data_labels(valid_s1, valid_s2, valid_s3,
                                                                                valid_labels)

    w_s1_c1, b_s1_c1 = weight_bias_variables([word_length, patch_size_s1_c1, 1, num_feat_s1_c1])
    w_s1_c2, b_s1_c2 = weight_bias_variables([1, patch_size_s1_c2, num_feat_s1_c1, num_feat_s1_c2])
    s1_c1_dim = int(math.floor((1 + num_words - patch_size_s1_c1) / pool_size_s1_c1))
    s1_c2_dim = int(math.floor((1 + s1_c1_dim - patch_size_s1_c2) / pool_size_s1_c2))
    flat_dim_s1 = s1_c2_dim * 1 * num_feat_s1_c2
    w_s1_fc1, b_s1_fc1 = weight_bias_variables([flat_dim_s1, num_feat_fc1])

    w_s2_c1, b_s2_c1 = weight_bias_variables([word_length, patch_size_s2_c1, 1, num_feat_s2_c1])
    w_s2_c2, b_s2_c2 = weight_bias_variables([1, patch_size_s2_c2, num_feat_s2_c1, num_feat_s2_c2])
    s2_c1_dim = int(math.floor((1 + num_words - patch_size_s2_c1) / pool_size_s2_c1))
    s2_c2_dim = int(math.floor((1 + s2_c1_dim - patch_size_s2_c2) / pool_size_s2_c2))
    flat_dim_s2 = s2_c2_dim * 1 * num_feat_s2_c2
    w_s2_fc1, b_s2_fc1 = weight_bias_variables([flat_dim_s2, num_feat_fc1])

    w_s3_c1, b_s3_c1 = weight_bias_variables([word_length, patch_size_s3_c1, 1, num_feat_s3_c1])
    w_s3_c2, b_s3_c2 = weight_bias_variables([1, patch_size_s3_c2, num_feat_s3_c1, num_feat_s3_c2])
    s3_c1_dim = int(math.floor((1 + num_words - patch_size_s3_c1) / pool_size_s3_c1))
    s3_c2_dim = int(math.floor((1 + s3_c1_dim - patch_size_s3_c2) / pool_size_s3_c2))
    flat_dim_s3 = s3_c2_dim * 1 * num_feat_s3_c2
    w_s3_fc1, b_s3_fc1 = weight_bias_variables([flat_dim_s3, num_feat_fc1])

    num_stacks = 3
    w_comb_c1, b_comb_c1 = weight_bias_variables([1, patch_size_comb_c1, 1, num_feat_comb_c1])
    comb_c1_dim = num_feat_comb_c1 * \
        int(math.floor((1 + (num_stacks * num_feat_fc1) - patch_size_comb_c1) / pool_size_comb_c1))
    w_comb_fc2, b_comb_fc2 = weight_bias_variables([comb_c1_dim, num_feat_fc2])
    w_comb_out, b_comb_out = weight_bias_variables([num_feat_fc2, num_labels])

    def stack_model(tf_s, w_s_c1, b_s_c1, w_s_c2, b_s_c2, w_s_fc1, b_s_fc1, pool_size_s_c1, pool_size_s_c2, flat_dim_s):
        h_s_c1 = tf.nn.relu(conv2d(tf_s, w_s_c1) + b_s_c1)
        h_s_p1 = max_pool(h_s_c1, pool_size_s_c1)
        h_s_n1 = tf.nn.local_response_normalization(h_s_p1)
        h_s_c2 = tf.nn.relu(conv2d(h_s_n1, w_s_c2) + b_s_c2)
        h_s_p2 = max_pool(h_s_c2, pool_size_s_c2)
        h_s_n2 = tf.nn.local_response_normalization(h_s_p2)
        h_s_n2_flat = tf.reshape(h_s_n2, [-1, flat_dim_s])
        return tf.matmul(h_s_n2_flat, w_s_fc1) + b_s_fc1

    def model(tf_s1, tf_s2, tf_s3, kp1=1.0, kp2=1.0):
        h_s1 = stack_model(tf_s1, w_s1_c1, b_s1_c1, w_s1_c2, b_s1_c2, w_s1_fc1, b_s1_fc1, pool_size_s1_c1,
                           pool_size_s1_c2, flat_dim_s1)
        h_s2 = stack_model(tf_s2, w_s2_c1, b_s2_c1, w_s2_c2, b_s2_c2, w_s2_fc1, b_s2_fc1, pool_size_s2_c1,
                           pool_size_s2_c2, flat_dim_s2)
        h_s3 = stack_model(tf_s3, w_s3_c1, b_s3_c1, w_s3_c2, b_s3_c2, w_s3_fc1, b_s3_fc1, pool_size_s3_c1,
                           pool_size_s3_c2, flat_dim_s3)
        tensor_print('h_s1', h_s1)
        h_comb_fc1 = tf.nn.relu(tf.concat(1, [h_s1, h_s2, h_s3]))
        h_comb_d1 = tf.nn.dropout(h_comb_fc1, kp1)
        tensor_print('h_comb_d1', h_comb_d1)
        h_comb_d1_reshaped = tf.reshape(h_comb_d1, [-1, 1, num_stacks * num_feat_fc1, 1])
        tensor_print('h_comb_d1_reshaped', h_comb_d1_reshaped)
        h_comb_c1 = tf.nn.relu(conv2d(h_comb_d1_reshaped, w_comb_c1) + b_comb_c1)
        tensor_print('h_comb_c1', h_comb_c1)
        h_comb_p1 = max_pool(h_comb_c1, pool_size_comb_c1)
        tensor_print('h_comb_p1', h_comb_p1)
        h_comb_n1 = tf.nn.local_response_normalization(h_comb_p1)
        tensor_print('h_comb_n1', h_comb_n1)
        print('comb_c1_dim', comb_c1_dim)
        h_comb_n1_flat = tf.reshape(h_comb_n1, [-1, comb_c1_dim])
        tensor_print('h_comb_n1_flat', h_comb_n1_flat)
        h_comb_fc2 = tf.nn.relu(tf.matmul(h_comb_n1_flat, w_comb_fc2) + b_comb_fc2)
        h_comb_d2 = tf.nn.dropout(h_comb_fc2, kp2)
        return tf.matmul(h_comb_d2, w_comb_out) + b_comb_out

    logits = model(tf_train_s1, tf_train_s2, tf_train_s3, keep_prob1, keep_prob2)
    tensor_print('logits', logits)

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(w_comb_out))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train_preds = tf.nn.softmax(logits)
    test_preds = tf.nn.softmax(model(tf_test_s1, tf_test_s2, tf_test_s3))
    valid_preds = tf.nn.softmax(model(tf_valid_s1, tf_valid_s2, tf_valid_s3))

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
        _, l, preds = session.run([optimizer, loss, train_preds], feed_dict=feed_dict)
        try:
            if step % 5000 == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(preds, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_preds.eval(), valid_labels))
        except KeyboardInterrupt:
            print('Test accuracy: %.1f%%' % accuracy(test_preds.eval(), test_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_preds.eval(), test_labels))