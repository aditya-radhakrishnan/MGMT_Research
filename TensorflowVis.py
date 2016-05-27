import tensorflow as tf
sess=tf.InteractiveSession()

with tf.named_scope('test'):
    v=tf.Variable([1,2,3])

sess.run(tf.initialize_all_variables())

print sess.run(v)
print v.eval()