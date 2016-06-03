import numpy as np
import tensorflow as tf
import math
from Gensim_Maker import GensimMaker
from SQL_Preprocessor_Wrapper_Version2 import SQLPreprocessorWrapperVersion2
import matplotlib.pyplot as plt

num_steps = 1000
step_print_size = 100
step_loss_size = 50
num_stacks = 3
word_length_s1 = 50
word_length_s2 = 20
word_length_s3 = 30

db_name = '/home/raditya/Documents/Patents/Databases/patent_database.db'
table_name = 'patent_info'
delimiter = '#!%^'
columns = ['abstract', 'inventors', 'assignee']
models = ['/home/raditya/Documents/Patents/Models/abstracts_model_50.txt',
          '/home/raditya/Documents/Patents/Models/inventors_model_20.txt',
          '/home/raditya/Documents/Patents/Models/assignees_model_30.txt']

make_models = [True, True, True]

ratios = [0.05, 0.02, 0.02]

gensim_makers = [GensimMaker(db_name, min_to_ignore=4, size=word_length_s1, sql_table_name=table_name,
                             column_name=columns[0], delimiter=delimiter, use_SQL_sentence_maker=True,
                             use_SQL_sentence_maker_for_texts=True),
                 GensimMaker(db_name, min_to_ignore=2, size=word_length_s2, sql_table_name=table_name,
                             column_name=columns[1], delimiter=delimiter, use_SQL_sentence_maker=True,
                             use_SQL_sentence_maker_for_texts=False),
                 GensimMaker(db_name, min_to_ignore=4, size=word_length_s3, sql_table_name=table_name,
                             column_name=columns[2], delimiter=delimiter, use_SQL_sentence_maker=True,
                             use_SQL_sentence_maker_for_texts=False)]


preprocessor_wrapper = SQLPreprocessorWrapperVersion2(db_name,
                                                      table_name,
                                                      columns,
                                                      models,
                                                      ratios,
                                                      delimiter,
                                                      gensim_makers,
                                                      need_to_make_models_bool_list=make_models)

preprocessor_wrapper.make_classification_dict('main_cpc_section')

num_words_s1 = preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence(processor_num=0)
num_words_s2 = preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence(processor_num=1)
num_words_s3 = preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence(processor_num=2)

train_s1 = preprocessor_wrapper.get_training_input_from_channel(channel_num=0)
train_s2 = preprocessor_wrapper.get_training_input_from_channel(channel_num=1)
train_s3 = preprocessor_wrapper.get_training_input_from_channel(channel_num=2)
train_labels = preprocessor_wrapper.get_training_labels()

valid_s1 = preprocessor_wrapper.get_validation_input_from_channel(channel_num=0)
valid_s2 = preprocessor_wrapper.get_validation_input_from_channel(channel_num=1)
valid_s3 = preprocessor_wrapper.get_validation_input_from_channel(channel_num=2)
valid_labels = preprocessor_wrapper.get_validation_labels()

test_s1 = preprocessor_wrapper.get_testing_input_from_channel(channel_num=0)
test_s2 = preprocessor_wrapper.get_testing_input_from_channel(channel_num=1)
test_s3 = preprocessor_wrapper.get_testing_input_from_channel(channel_num=2)
test_labels = preprocessor_wrapper.get_testing_labels()

num_train_docs = train_labels.shape[0]
num_test_docs = test_labels.shape[0]
num_valid_docs = valid_labels.shape[0]
num_labels = train_labels.shape[1]

# Constants for Abstracts stack
patch_size_s1_c1 = 10
pool_size_s1_c1 = 4
num_feat_s1_c1 = 64

patch_size_s1_c2 = 3
pool_size_s1_c2 = 2
num_feat_s1_c2 = 32

num_feat_s1_fc1 = 16

# Constants for Inventors stack
patch_size_s2_c1 = 3
pool_size_s2_c1 = 2
num_feat_s2_c1 = 16

patch_size_s2_c2 = 3
pool_size_s2_c2 = 1
num_feat_s2_c2 = 8

num_feat_s2_fc1 = 8

# Constants for Assignees stack
patch_size_s3_c1 = 2
pool_size_s3_c1 = 2
num_feat_s3_c1 = 8

patch_size_s3_c2 = 3
pool_size_s3_c2 = 1
num_feat_s3_c2 = 4

num_feat_s3_fc1 = 4

# Combined network constants
num_feat_comb_fc1 = num_feat_s1_fc1 + num_feat_s2_fc1 + num_feat_s3_fc1
keep_prob1 = 0.5

patch_size_comb_c1 = 2
pool_size_comb_c1 = 2
num_feat_comb_c1 = 16

num_feat_fc2 = 16
keep_prob2 = 0.5

batch_size = int(math.floor(num_train_docs / 25))
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
    def gen_tf_data_labels(s1, s2, s3, labels):
        return tf.constant(s1, dtype=tf.float32), tf.constant(s2, dtype=tf.float32), \
               tf.constant(s3, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)

    tf_train_s1 = tf.placeholder(dtype=tf.float32)
    tf_train_s2 = tf.placeholder(dtype=tf.float32)
    tf_train_s3 = tf.placeholder(dtype=tf.float32)
    tf_train_labels = tf.placeholder(dtype=tf.float32)

    tf_test_s1, tf_test_s2, tf_test_s3, tf_test_labels = gen_tf_data_labels(test_s1, test_s2, test_s3, test_labels)
    tf_valid_s1, tf_valid_s2, tf_valid_s3, tf_valid_labels = gen_tf_data_labels(valid_s1, valid_s2, valid_s3,
                                                                                valid_labels)

    w_s1_c1, b_s1_c1 = weight_bias_variables([word_length_s1, patch_size_s1_c1, 1, num_feat_s1_c1])
    w_s1_c2, b_s1_c2 = weight_bias_variables([1, patch_size_s1_c2, num_feat_s1_c1, num_feat_s1_c2])
    s1_c1_dim = int(math.floor((1 + num_words_s1 - patch_size_s1_c1) / pool_size_s1_c1))
    s1_c2_dim = int(math.floor((1 + s1_c1_dim - patch_size_s1_c2) / pool_size_s1_c2))
    flat_dim_s1 = s1_c2_dim * 1 * num_feat_s1_c2
    w_s1_fc1, b_s1_fc1 = weight_bias_variables([flat_dim_s1, num_feat_s1_fc1])

    w_s2_c1, b_s2_c1 = weight_bias_variables([word_length_s2, patch_size_s2_c1, 1, num_feat_s2_c1])
    w_s2_c2, b_s2_c2 = weight_bias_variables([1, patch_size_s2_c2, num_feat_s2_c1, num_feat_s2_c2])
    s2_c1_dim = int(math.floor((1 + num_words_s2 - patch_size_s2_c1) / pool_size_s2_c1))
    s2_c2_dim = int(math.floor((1 + s2_c1_dim - patch_size_s2_c2) / pool_size_s2_c2))
    flat_dim_s2 = s2_c2_dim * 1 * num_feat_s2_c2
    w_s2_fc1, b_s2_fc1 = weight_bias_variables([flat_dim_s2, num_feat_s2_fc1])

    w_s3_c1, b_s3_c1 = weight_bias_variables([word_length_s3, patch_size_s3_c1, 1, num_feat_s3_c1])
    w_s3_c2, b_s3_c2 = weight_bias_variables([1, patch_size_s3_c2, num_feat_s3_c1, num_feat_s3_c2])
    s3_c1_dim = int(math.floor((1 + num_words_s3 - patch_size_s3_c1) / pool_size_s3_c1))
    s3_c2_dim = int(math.floor((1 + s3_c1_dim - patch_size_s3_c2) / pool_size_s3_c2))
    flat_dim_s3 = s3_c2_dim * 1 * num_feat_s3_c2
    w_s3_fc1, b_s3_fc1 = weight_bias_variables([flat_dim_s3, num_feat_s3_fc1])

    w_comb_c1, b_comb_c1 = weight_bias_variables([1, patch_size_comb_c1, 1, num_feat_comb_c1])
    comb_c1_dim = num_feat_comb_c1 * \
        int(math.floor((1 + num_feat_comb_fc1 - patch_size_comb_c1) / pool_size_comb_c1))
    w_comb_fc2, b_comb_fc2 = weight_bias_variables([comb_c1_dim, num_feat_fc2])
    w_comb_out, b_comb_out = weight_bias_variables([num_feat_fc2, num_labels])

    def stack_model(tf_s, w_s_c1, b_s_c1, w_s_c2, b_s_c2, w_s_fc1, b_s_fc1, pool_size_s_c1, pool_size_s_c2, flat_dim_s):
        # tensor_print('tf_s', tf_s)
        h_s_c1 = tf.nn.relu(conv2d(tf_s, w_s_c1) + b_s_c1)
        # tensor_print('h_s_c1', h_s_c1)
        h_s_p1 = max_pool(h_s_c1, pool_size_s_c1)
        # tensor_print('h_s_p1', h_s_p1)
        h_s_n1 = tf.nn.local_response_normalization(h_s_p1)
        # tensor_print('h_s_n1', h_s_n1)
        h_s_c2 = tf.nn.relu(conv2d(h_s_n1, w_s_c2) + b_s_c2)
        # tensor_print('h_s_c2', h_s_c2)
        h_s_p2 = max_pool(h_s_c2, pool_size_s_c2)
        # tensor_print('h_s_p2', h_s_p2)
        h_s_n2 = tf.nn.local_response_normalization(h_s_p2)
        # tensor_print('h_s_n2', h_s_n2)
        h_s_n2_flat = tf.reshape(h_s_n2, [-1, flat_dim_s])
        # tensor_print('h_s_n2_flat', h_s_n2_flat)
        return tf.matmul(h_s_n2_flat, w_s_fc1) + b_s_fc1

    def model(tf_s1, tf_s2, tf_s3, kp1=1.0, kp2=1.0):
        h_s1 = stack_model(tf_s1, w_s1_c1, b_s1_c1, w_s1_c2, b_s1_c2, w_s1_fc1, b_s1_fc1, pool_size_s1_c1,
                           pool_size_s1_c2, flat_dim_s1)
        h_s2 = stack_model(tf_s2, w_s2_c1, b_s2_c1, w_s2_c2, b_s2_c2, w_s2_fc1, b_s2_fc1, pool_size_s2_c1,
                           pool_size_s2_c2, flat_dim_s2)
        h_s3 = stack_model(tf_s3, w_s3_c1, b_s3_c1, w_s3_c2, b_s3_c2, w_s3_fc1, b_s3_fc1, pool_size_s3_c1,
                           pool_size_s3_c2, flat_dim_s3)
        # tensor_print('h_s1', h_s1)
        h_comb_fc1 = tf.nn.relu(tf.concat(1, [h_s1, h_s2, h_s3]))
        h_comb_d1 = tf.nn.dropout(h_comb_fc1, kp1)
        # tensor_print('h_comb_d1', h_comb_d1)
        h_comb_d1_reshaped = tf.reshape(h_comb_d1, [-1, 1, num_feat_comb_fc1, 1])
        # tensor_print('h_comb_d1_reshaped', h_comb_d1_reshaped)
        h_comb_c1 = tf.nn.relu(conv2d(h_comb_d1_reshaped, w_comb_c1) + b_comb_c1)
        # tensor_print('h_comb_c1', h_comb_c1)
        h_comb_p1 = max_pool(h_comb_c1, pool_size_comb_c1)
        # tensor_print('h_comb_p1', h_comb_p1)
        h_comb_n1 = tf.nn.local_response_normalization(h_comb_p1)
        # tensor_print('h_comb_n1', h_comb_n1)
        # print('comb_c1_dim', comb_c1_dim)
        h_comb_n1_flat = tf.reshape(h_comb_n1, [-1, comb_c1_dim])
        # tensor_print('h_comb_n1_flat', h_comb_n1_flat)
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
    steps = np.arange(0, num_steps, step_loss_size)
    losses = np.array([0])
    valids = np.array([0])
    for step in range(num_steps):
        batch_indices = np.random.choice(num_train_docs, size=batch_size, replace=False)
        batch_s1 = train_s1[batch_indices, :, :, :]
        batch_s2 = train_s2[batch_indices, :, :, :]
        batch_s3 = train_s3[batch_indices, :, :, :]
        batch_labels = train_labels[batch_indices, :]
        feed_dict = {tf_train_s1: batch_s1, tf_train_s2: batch_s2, tf_train_s3: batch_s3, tf_train_labels: batch_labels}
        _, l, preds = session.run([optimizer, loss, train_preds], feed_dict=feed_dict)
        try:
            if step % step_loss_size == 0:
                losses = np.concatenate((losses, np.array([l])))
                valids = np.concatenate((valids, np.array([accuracy(valid_preds.eval(), valid_labels)])))
            if step % step_print_size == 0:
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(preds, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(valid_preds.eval(), valid_labels))
        except KeyboardInterrupt:
            print('Test accuracy: %.1f%%' % accuracy(test_preds.eval(), test_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_preds.eval(), test_labels))
    losses = losses[1:]
    valids = valids[1:]
    plt.figure(1)

    plt.subplot(211)
    plt.xlabel('Step Number')
    plt.ylabel('Loss')
    plt.plot(steps, losses)

    plt.subplot(212)
    plt.xlabel('Step Number')
    plt.ylabel('Validation Accuracy')
    plt.plot(steps, valids)

    plt.show()
