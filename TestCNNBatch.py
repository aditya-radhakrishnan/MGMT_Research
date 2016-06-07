import numpy as np
import tensorflow as tf

# Count of training, test and validation sets
num_train_docs = 2000
num_test_docs = 500
num_valid_docs = 500
num_labels = 50
# The dimensions of each document (vector_length, num_words)
doc_dims = (20, 500)

def gen_examples(num_docs):
    examples = np.random.random_sample(doc_dims)
    for i in range(num_docs - 1):
        new_example = np.random.random_sample(doc_dims)
        examples = np.concatenate((examples, new_example), axis=0)
    return np.reshape(examples, (-1, doc_dims[0], doc_dims[1], 1))

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

# Neural network constants
batch_size = 5
num_words = 2
num_features = 8
num_fully_connected_neurons = 16
beta = 0.01
# depth = num_words * doc_dims[0]
stride_length = 1
# padding = True

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

'''
def max_pool(x, pool_size):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
'''

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

graph = tf.Graph()

with graph.as_default():
    def gen_tf_data_labels(data, labels):
        return tf.constant(data, dtype=tf.float32), tf.constant(labels, dtype=tf.float32)

    #tf_train_data, tf_train_labels = gen_tf_data_labels(train_data, train_labels)
    tf_train_data = tf.placeholder(dtype=tf.float32)
    tf_train_labels = tf.placeholder(dtype=tf.float32)
    tf_test_data, tf_test_labels = gen_tf_data_labels(test_data, test_labels)
    tf_valid_data, tf_valid_labels = gen_tf_data_labels(valid_data, valid_labels)

    # Weights are of the shape [vector_length, patch_size, num_features], biases are of the shape [num_features]
    weights_conv = weight_variable([doc_dims[0], num_words, 1, num_features])
    biases_conv = bias_variable([num_features])

    # Weights and biases for full connected layer
    size = doc_dims[0] * doc_dims[1] * num_features
    weights_full_conn = weight_variable([size, num_fully_connected_neurons])
    print(weights_full_conn.get_shape()[0])
    biases_full_conn = bias_variable([num_fully_connected_neurons])

    # Weights and biases for softmax layer
    weights_output = weight_variable([num_fully_connected_neurons, num_labels])
    biases_output = bias_variable([num_labels])

    def model(tf_reshaped_data):
        h_conv = tf.nn.relu(conv2d(tf_reshaped_data, weights_conv) + biases_conv)
        h_conv_flat = tf.reshape(h_conv, [-1, size])
        h_full_conn = tf.nn.relu(tf.matmul(h_conv_flat, weights_full_conn) + biases_full_conn)
        return tf.matmul(h_full_conn, weights_output) + biases_output

    logits = model(tf_train_data)

    # Using gradient descent to minimize loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
    loss = tf.reduce_mean(loss + beta * tf.nn.l2_loss(weights_output))
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(model(tf_test_data))
    valid_prediction = tf.nn.softmax(model(tf_valid_data))

num_steps = 1000

with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    for step in range(num_steps):
        # offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        numbers = np.random.choice(20, size=batch_size, replace=False)
        #batch_data = train_data[offset:(offset + batch_size), :, :, :]
        # batch_labels = train_labels[offset:(offset + batch_size), :]
        batch_data = train_data[numbers, :, :, :]
        batch_labels = train_labels[numbers, :]
        feed_dict = {tf_train_data: batch_data, tf_train_labels: batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Indices', numbers)
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels))
    print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))