#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import sys

from plaid_data_setup import get_input_len, get_labels_len, train_cycle_nn, gen_data, train_cycle_sequence_nn, train_cycle_hierarchy_nn

# Parametsr for synthetic data
batch_size = 10
trace_prediciton_length = 7
padded_trace_prediction_length = trace_prediciton_length + 10 # In my head this should be the timestamp?
num_labels = 11

#LSTM parameters.
num_units = 22
learning_rate = .005

# Artificial data I created just to do some basic tests
synthetic_data = np.random.rand(batch_size, trace_prediciton_length, num_labels)
padding_shape = ((0,0), (0,10), (0,0)) # Only pad along with trace_prediction_length axis
synthetic_data = np.pad(synthetic_data, padding_shape, "constant", constant_values=(0,0)) # Zero pad our synethtic data laong withe trace_predition_length axis
synthetic_correct = np.zeros(shape=(batch_size, num_labels), dtype=int)

for x in synthetic_correct:
	x[5] = 1

synth_sequence_lengths = []
for i in range(len(synthetic_data)):
	synth_sequence_lengths.append(int(np.argmin(synthetic_data[i]) / num_labels))



inputs = tf.placeholder(tf.float32, (None, None, num_labels))
correct_labels = tf.placeholder(tf.float32, (None, num_labels))
seqlen = tf.placeholder(tf.int32, None) # The true sequence lengths of the inputs. THe actual inputs should be zero-padded such that they are all the same length.

cell = tf.contrib.rnn.BasicLSTMCell(num_units) # Single layer LSTM

outputs, state = tf.nn.dynamic_rnn(cell=cell,
								 inputs=inputs,
								 sequence_length=seqlen, # Tensorflow uses this to automatically mask out the zero padding
								 dtype=tf.float32)

last_output = tf.gather_nd(outputs, tf.stack([tf.range(tf.shape(outputs)[0]), seqlen-1], axis=1)) # This gets the last output before zero padding

# This variable_scope stuff seems unneccessary here, but seems like a good habit for tensorflow. Basically helps you manage variables when there are multiple neural networks
with tf.variable_scope('softmax'):
	W = tf.get_variable('W', [num_units, num_labels])
	b = tf.get_variable('b', [num_labels], initializer = tf.constant_initializer(0.0))

logits = tf.matmul(last_output, W) + b
softmax = tf.nn.softmax(logits)

# Copied this shit from our other networks
cost = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_labels)
loss = tf.reduce_mean(cost)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss)

# Eval Shit
predictions = tf.argmax(softmax, 1)
pred_scores = tf.reduce_max(softmax,1)
correct_pred = tf.equal(predictions, tf.argmax(correct_labels, 1)) # check the index with the largest value
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) # percentage of traces that were correct

lstm_graph = tf.get_default_graph()
evaluation_args = [loss, accuracy, predictions, pred_scores, correct_pred]
train_cycle_sequence_nn(lstm_graph, inputs, correct_labels, seqlen, train_op, evaluation_args, gen_data(), 'dense_single_layer')


"""

test_args = [inputs, outputs, last_output]

with tf.Session() as sess:
	sess.run(init)

	step = 0

	while True:
		step += 1



	test_in, test_puts, test_last_put = sess.run(test_args, feed_dict={inputs: synthetic_data, correct_labels: synthetic_correct, seqlen: synth_sequence_lengths})
	print(test_puts)
	print(tf.shape(test_puts))
	print(len(test_puts))
	print("--------------------------------")
	print(test_puts[0])
	print(tf.shape(test_puts[0]))
	print(len(test_puts[0]))
	print("+++++++++++++++++++++++++++++++")
	print(test_last_put)
	print(test_puts.shape)
	print(test_last_put.shape)
	#print(test_last_put)

	#print(test_last_put[0])
	#print(tf.shape(test_last_put[0]))
"""
