import tensorflow as tf

####################################################################################
## Utilities
##
## aka, things that could probably move into a generic DeepIoT module

# dropOut_prun
#
# Oy. Vey. Okay, the code is left like this because it matches the original, but
# here's what I think is going on. First, recall that these probabilities will
# be fed into a Bernoulli distribution, aka 0->drop, 1->keep, .5->50/50 drop.
#
#   During the compress phase (the false_fn case), we sweep prun_thres from 0->1
#   in modest steps. This returns a vector of actual drop probabilities based on
#   the stored drop probability and an increasing likelihood of being "decayed"
#   by 0.5 (that is, as prun_thres rises, nodes are pruned more aggressively, as
#   their chance not being dropped cuts in half)
#
#   In the fine-tuning phase (the true_fn case), prun_thres is pinned at some
#   final value (e.g. running their example network, prun_thres made it to
#   0.825 when the compression ratio target was reached). Now, for nodes with a
#   dropout probability less than prun_thres, zero the probability - aka drop
#   that node - and for the rest, use the learned dropout probability.
#
#   Aha! The whole point of the fine-tuning phase, then, is to re-train the
#   final network using the learned dropout probabilities. It's all coming
#   together.
def dropout_prune(drop_prob, prun_thres, compress_done):
    base_prob = 0.5 # "decay factor" γ from eq (10)
    pruned_drop_prob = tf.cond(
            compress_done > 0.5, # 0.5 is meaningless, just 0 or 1
            lambda: tf.where(tf.less(drop_prob, prun_thres), tf.zeros_like(drop_prob), drop_prob), # true_fn
            lambda: tf.where(tf.less(drop_prob, prun_thres), drop_prob * base_prob, drop_prob),    # false_fn
            )
    return pruned_drop_prob

# Given a dictionary of "layer_name: dropout_probabilities", return a dictionary
# of "layer_name: <tf.fn>[count_of_nodes_not_pruned]" (those "left")
def count_prun(layer_to_probability, prun_thres):
    # "left_num" is the number of nodes left in each layer
    left_num_dict = {}
    for layer_name, prob_list in layer_to_probability.items():
        pruned_idt = tf.where(tf.less(prob_list, prun_thres), tf.zeros_like(prob_list), tf.ones_like(prob_list))
        left_num = tf.reduce_sum(pruned_idt)
        left_num_dict[layer_name] = left_num
    return left_num_dict

# In concert with 'count_prun', execute to get counts
#
# Returns "layer_name: count_of_nodes_not_pruned" (those "left")
def gen_cur_prun(sess, left_num_dict):
    cur_left_num = {}
    for layer_name, tf_fn in left_num_dict.items():
        cur_left_num[layer_name] = sess.run(tf_fn)
    return cur_left_num

# Given two "layer_name: nodes" mappings, see how much better we're doing
#
# XXX Not generic, assumes knowledge of how to interperet layer names
# XXX Fuck it, makin it worse with n_input, n_labels
def compress_ratio(cur_left_num, org_dim_dict, n_input, n_labels):
    original_size   = 0
    compressed_size = 0

    # Weights: input -> hidden
    original_size   += n_input * org_dim_dict['fc1']
    compressed_size += n_input * cur_left_num['fc1']
    # Weights: hidden -> output
    original_size   += org_dim_dict['fc1'] * n_labels
    compressed_size += cur_left_num['fc1'] * n_labels
    # Biases: hidden
    original_size   += org_dim_dict['fc1']
    compressed_size += cur_left_num['fc1']
    # Biases: output
    original_size   += n_labels
    compressed_size += n_labels

    percent = 100*(compressed_size/original_size)

    print("  Original Size:", original_size, "Compressed Size:", compressed_size, "Percent:", percent)

    return percent

##
## Utilities End
####################################################################################

