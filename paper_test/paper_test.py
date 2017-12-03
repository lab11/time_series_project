#! /usr/bin/env python3

# NOTE: In order to make the code simple, we rewrite x * W_1 + b_1 = x' * W_1'
# where x' = [x | 1] and W_1' is the matrix W_1 appended with a new row with elements b_1's.
# Similarly, for h * W_2 + b_2
import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
tf.set_random_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# load and shuffle data
data = np.load("../plaid_data/traces_bundle.npy")
np.random.shuffle(data)
Data = data[:, 0:-2]
Labels = data[:,-1]
house_ids = data[:,-2]

# Config:
n_hidden    = 30
n_input     = 2*2*500#int(2.52E3/60)
n_classes   = 11
learning_rate = 0.1
n_steps     = 500
display_step= 100
batch_size  = 50

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

weights = {
    'h1':  tf.Variable(tf.random_normal([n_input, n_hidden])),
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes])),
}
biases = {
    'b1':   tf.Variable(tf.random_normal([n_hidden])),
    'out':  tf.Variable(tf.random_normal([n_classes])),
}

def neural_net(x):
    # hidden fully connected layer
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # output fully connected layer, neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:

    sess.run(init)

    for step in range(1, n_steps+1):
        batch_nums = np.random.choice(Data.shape[0], batch_size)
        test_nums = np.random.choice(Data.shape[0], 2000)

        sess.run(train_op, feed_dict = {X: Data[batch_nums], Y: Labels[batch_nums]})
        if step % display_step == 0 or step == 1:
            # calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: Data[test_nums], Y: Labels[test_nums]})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished")


