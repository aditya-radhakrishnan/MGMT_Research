import numpy as np
import tensorflow as tf
from Preprocessing import Preprocessing

processor = Preprocessing('/home/raditya/Documents/untitled folder/All_Data',
                          '/home/raditya/Documents/untitled folder/EarningsCallModel.txt', need_to_create_model=True)

# Count of training, test and validation sets
num_train_docs = 6
num_test_docs = 3
num_valid_docs = 3
num_labels = 4
# Document dimensions
dropout_keep_prob = 0.8
word_length = processor.gensim_maker_obj.get_dimension_of_a_word()
num_words = processor.max_num_words
'''
def gen_examples(num_docs):
    examples = np.random.random_sample((word_length, num_words))
    for i in range(num_docs - 1):
        new_example = np.random.random_sample((word_length, num_words))
        examples = np.concatenate((examples, new_example), axis=0)
    return np.reshape(examples, (-1, word_length, num_words, 1))

def gen_labels(num_docs):
    index = np.random.randint(0, num_labels)
    labels = np.zeros((1, num_labels))
    labels[0][index] = 1
    for i in range(num_docs - 1):
        index = np.random.randint(0, num_labels)
        row = np.zeros((1, num_labels))
        row[0][index] = 1
        labels = np.concatenate((labels, row), axis=0)
    return labels

def gen_data(num_docs):
    return gen_examples(num_docs), gen_labels(num_docs)

train_data, train_labels = gen_data(num_train_docs)
test_data, test_labels = gen_data(num_test_docs)
valid_data, valid_labels = gen_data(num_valid_docs)
'''

train_data, train_labels = processor.get_data_set(num_train_docs,
                                                  '/home/raditya/Documents/untitled folder/Training_Data')
test_data, test_labels = processor.get_data_set(num_test_docs,
                                                '/home/raditya/Documents/untitled folder/Test_Data')
valid_data, valid_labels = processor.get_data_set(num_valid_docs,
                                                  '/home/raditya/Documents/untitled folder/Validation_Data')

# Neural network constants
batch_size = 6
patch_sizes1 = [2, 5]
patch_sizes2 = [2, 5]
num_features1 = [2, 2]
num_features2 = [2, 2]
num_full_conn = 4
learning_rate = 0.1
beta = 0.01

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

graph = tf.Graph()

with graph.as_default():
    def gen_tf_data_labels(data, labels):
        return tf.constant(data, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)

    tf_train_data = tf.placeholder(dtype=tf.float32)
    tf_train_labels = tf.placeholder(dtype=tf.float32)
    tf_test_data, tf_test_labels = gen_tf_data_labels(test_data, test_labels)
    tf_valid_data, tf_valid_labels = gen_tf_data_labels(valid_data, valid_labels)

    def layer_weights_biases(num_features_in, patch_sizes, num_features_out):
        assert len(patch_sizes) == len(num_features_out)
        weights = []
        biases = []
        for i in range(len(num_features_out)):
            weight_layer = []
            for j in range(len(num_features_in)):
                weight = weight_variable([word_length, patch_sizes[i], num_features_in[j], num_features_out[i]])
                weight_layer.append(weight)
            bias = bias_variable([num_features_out[i]])
            weights.append(weight_layer)
            biases.append(bias)
        return weights, biases

    def activations(inputs, weights, biases):
        assert len(weights) == len(biases)
        assert len(weights[0]) == len(inputs)
        outputs = []
        for i in range(len(weights)):
            for j in range(len(weights[0])):
                h = tf.nn.relu(conv2d(inputs[j], weights[i][j]) + biases[i])
                outputs.append(h)
        return outputs

    def full_conn_weights_biases(inputs):
        weights = []
        for i in range(len(inputs)):
            dim = inputs[i].get_shape()[3]
            input_size = word_length * num_words * dim.value
            weight = weight_variable([input_size, num_full_conn])
            weights.append(weight)
        return weights, bias_variable([num_full_conn])

    def reshape(inputs):
        reshaped_inputs = []
        for i in range(len(inputs)):
            dim = inputs[i].get_shape()[3]
            input_size = word_length * num_words * dim.value
            reshaped_input = tf.reshape(inputs[i], [-1, input_size])
            reshaped_inputs.append(reshaped_input)
        return reshaped_inputs

    def full_conn_calc(inputs, weights, bias):
        hyp = tf.matmul(inputs[0], weights[0]) + bias
        for i in range(1, len(inputs)):
            hyp += tf.matmul(inputs[i], weights[i]) + bias
        return tf.nn.relu(hyp)

    def model(tf_reshaped_data, keep_prob=tf.constant(1.0)):
        w_conv1, b_conv1 = layer_weights_biases([1], patch_sizes1, num_features1)
        conv_layer1 = activations([tf_reshaped_data], w_conv1, b_conv1)

        w_conv2, b_conv2 = layer_weights_biases(num_features1, patch_sizes2, num_features2)
        conv_layer2 = activations(conv_layer1, w_conv2, b_conv2)

        w_full_conn, b_full_conn = full_conn_weights_biases(conv_layer2)
        hyp_full_conn = full_conn_calc(reshape(conv_layer2), w_full_conn, b_full_conn)

        hyp_drop = tf.nn.dropout(hyp_full_conn, keep_prob)

        weights_final = weight_variable([num_full_conn, num_labels])
        biases_final = bias_variable([num_labels])
        return weights_final, tf.matmul(hyp_drop, weights_final) + biases_final


    weights_output, logits = model(tf_train_data, keep_prob=tf.constant(dropout_keep_prob))

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(weights_output))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    _, test_intermed = model(tf_test_data)
    test_prediction = tf.nn.softmax(test_intermed)
    _, valid_intermed = model(tf_valid_data)
    valid_prediction = tf.nn.softmax(valid_intermed)

num_steps = 20

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        batch_indices = np.random.choice(num_train_docs, size=batch_size, replace=False)
        batch_data = train_data[batch_indices, :, :, :]
        batch_labels = train_labels[batch_indices, :]
        feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 1 == 0):
            print('Indices', batch_indices)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))