import numpy as np
import tensorflow as tf
import math
from Gensim_Maker import GensimMaker
from SQL_Preprocessor_Wrapper import SQLPreprocessorWrapper
import matplotlib.pyplot as plt

num_steps = 5000
step_print_size = 1000
step_loss_size = 100
num_stacks = 3
word_length_s1 = 30
word_length_s2 = 10
word_length_s3 = 20

db_name = '/home/raditya/Documents/Patents/Databases/patent_database.db'
input_tables = ['abstracts', 'inventors', 'assignees']
models = ['/home/raditya/Documents/Patents/Models/abstracts_model_30.txt',
          '/home/raditya/Documents/Patents/Models/inventors_model_10.txt',
          '/home/raditya/Documents/Patents/Models/assignees_model_20.txt']
output_tables = ['organized_abstracts_30', 'organized_inventors_10', 'assignees_model_20']

make_models = [False, False, False]
make_tables = [False, False, False]

gensim_makers = [GensimMaker(db_name, min_to_ignore=4, size=word_length_s1, sql_table_name=input_tables[0],
                             use_SQL_sentence_maker=True, use_SQL_sentence_maker_for_texts=True),
                 GensimMaker(db_name, min_to_ignore=2, size=word_length_s2, sql_table_name=input_tables[1],
                             use_SQL_sentence_maker=True),
                 GensimMaker(db_name, min_to_ignore=4, size=word_length_s3, sql_table_name=input_tables[2],
                             use_SQL_sentence_maker=True, use_SQL_sentence_maker_for_texts=True)]

ratios = [0.7, 0.15, 0.15]
preprocessor_wrapper = SQLPreprocessorWrapper(db_name, input_tables, output_tables, models, ratios,
                                              list_of_gensim_makers=gensim_makers,
                                              need_to_make_models_bool_list=make_models,
                                              need_to_make_tables_bool_list=make_tables)
preprocessor_wrapper.make_classification_dict('classifications')

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


def extract(function):
    output = []
    for i in range(num_stacks):
        output.append(function(i))
    return output

word_lengths = [word_length_s1, word_length_s2, word_length_s3]

num_words_list = extract(preprocessor_wrapper.get_maximum_number_of_words_in_a_sentence)
train_list = extract(preprocessor_wrapper.get_training_input_from_channel)
valid_list = extract(preprocessor_wrapper.get_validation_input_from_channel)
test_list = extract(preprocessor_wrapper.get_training_input_from_channel)
train_labels = preprocessor_wrapper.get_training_labels()
valid_labels = preprocessor_wrapper.get_validation_labels()
test_labels = preprocessor_wrapper.get_testing_labels()

num_train_docs = train_labels.shape[0]
num_test_docs = test_labels.shape[0]
num_valid_docs = valid_labels.shape[0]
num_labels = train_labels.shape[1]

patch_sizes_s_c1 = [patch_size_s1_c1, patch_size_s2_c1, patch_size_s3_c1]
pool_sizes_s_c1 = [pool_size_s1_c1, pool_size_s2_c1, pool_size_s3_c1]
num_feat_s_c1 = [num_feat_s1_c1, num_feat_s2_c1, num_feat_s3_c1]

patch_sizes_s_c2 = [patch_size_s1_c2, patch_size_s2_c2, patch_size_s3_c2]
pool_sizes_s_c2 = [pool_size_s1_c2, pool_size_s2_c2, pool_size_s3_c2]
num_feat_s_c2 = [num_feat_s1_c2, num_feat_s2_c2, num_feat_s3_c2]

num_feat_s_fc1 = [num_feat_s1_fc1, num_feat_s2_fc1, num_feat_s3_fc1]

num_feat_comb_fc1 = sum(num_feat_s_fc1)
keep_prob1 = 0.5

patch_size_comb_c1 = 2
pool_size_comb_c1 = 2
num_feat_comb_c1 = 16

num_feat_fc2 = 16
keep_prob2 = 0.5

batch_size = 10
beta = 0.01

ones = []
for a in range(num_stacks):
    ones.append(1)


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

    def gen_tf_data_labels(data_list, labels, function):
        output = []
        for i in range(len(data_list)):
            output.append(function(data_list[i], dtype=tf.float32))
        return output, function(labels, dtype=tf.float32)

    tf_train_list, tf_train_labels = gen_tf_data_labels(train_list, train_labels, tf.placeholder)
    tf_test_list, tf_test_labels = gen_tf_data_labels(test_list, test_labels, tf.constant)
    tf_valid_list, tf_valid_labels = gen_tf_data_labels(valid_list, valid_labels, tf.constant)

    def weights_biases_transform(first_dim, second_dim, third_dim, fourth_dim):
        weights = []
        biases = []
        for i in range(num_stacks):
            weight, bias = weight_bias_variables([first_dim[i], second_dim[i], third_dim[i], fourth_dim[i]])
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    def weights_biases_transform_small(first_dim, second_dim):
        weights = []
        biases = []
        for i in range(num_stacks):
            weight, bias = weight_bias_variables([first_dim[i], second_dim[i]])
            weights.append(weight)
            biases.append(bias)
        return weights, biases

    w_s_c1, b_s_c1 = weights_biases_transform(word_lengths, patch_sizes_s_c1, ones, num_feat_s_c1)
    w_s_c2, b_s_c2 = weights_biases_transform(ones, patch_sizes_s_c2, num_feat_s_c1, num_feat_s_c2)

    def flat_dims():
        outputs = []
        for i in range(num_stacks):
            s_c1_dim = int(math.floor((1 + num_words_list[i] - patch_sizes_s_c1[i]) / pool_sizes_s_c1[i]))
            s_c2_dim = int(math.floor((1 + s_c1_dim - patch_sizes_s_c2[i]) / pool_sizes_s_c2[i]))
            flat_dim_s = s_c2_dim * num_feat_s_c2[i]
            outputs.append(flat_dim_s)
        return outputs

    flat_dims = flat_dims()
    w_s_fc1, b_s_fc1 = weights_biases_transform_small(flat_dims, num_feat_s_fc1)

    w_comb_c1, b_comb_c1 = weight_bias_variables([1, patch_size_comb_c1, 1, num_feat_comb_c1])
    comb_c1_dim = num_feat_comb_c1 * \
        int(math.floor((1 + num_feat_comb_fc1 - patch_size_comb_c1) / pool_size_comb_c1))
    w_comb_fc2, b_comb_fc2 = weight_bias_variables([comb_c1_dim, num_feat_fc2])
    w_comb_out, b_comb_out = weight_bias_variables([num_feat_fc2, num_labels])

    def stack_model(tf_s, index):
        # tensor_print('tf_s', tf_s)
        h_s_c1 = tf.nn.relu(conv2d(tf_s, w_s_c1[index]) + b_s_c1[index])
        # tensor_print('h_s_c1', h_s_c1)
        h_s_p1 = max_pool(h_s_c1, pool_sizes_s_c1[index])
        # tensor_print('h_s_p1', h_s_p1)
        h_s_n1 = tf.nn.local_response_normalization(h_s_p1)
        # tensor_print('h_s_n1', h_s_n1)
        h_s_c2 = tf.nn.relu(conv2d(h_s_n1, w_s_c2[index]) + b_s_c2[index])
        # tensor_print('h_s_c2', h_s_c2)
        h_s_p2 = max_pool(h_s_c2, pool_sizes_s_c2[index])
        # tensor_print('h_s_p2', h_s_p2)
        h_s_n2 = tf.nn.local_response_normalization(h_s_p2)
        # tensor_print('h_s_n2', h_s_n2)
        h_s_n2_flat = tf.reshape(h_s_n2, [-1, flat_dims[index]])
        # tensor_print('h_s_n2_flat', h_s_n2_flat)
        return tf.matmul(h_s_n2_flat, w_s_fc1[index]) + b_s_fc1[index]

    def model(tf_data, kp1=1.0, kp2=1.0):
        h_comb = stack_model(tf_data[0], 0)
        for i in range(1, num_stacks):
            tf.concat(1, [h_comb, stack_model(tf_data[i], i)])
        h_comb_fc1 = tf.nn.relu(h_comb)
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

    logits = model(tf_train_list, keep_prob1, keep_prob2)
    # tensor_print('logits', logits)

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(w_comb_out))
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    train_preds = tf.nn.softmax(logits)
    test_preds = tf.nn.softmax(model(tf_test_list))
    valid_preds = tf.nn.softmax(model(tf_valid_list))

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
