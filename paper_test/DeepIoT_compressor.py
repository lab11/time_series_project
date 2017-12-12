import tensorflow as tf
import numpy as np
import pprint

import DeepIoT_utilities

####################################################################################
## Compressor Part Start
##

# A copy of the original function. Its goal is to convert the weights from a
# hidden layer into something that's amenable to feeding into the meta-training
# RNN. It's network-specific as-written, using the layer_name to determine what
# to do. In the example network, DeepIoT is operating on some CNN layers and
# some GRU-based RNN layers (which have 'cell' in their layer_name's).
#
# In our case, the fully connected weights are already in pretty good shape.
def concat_weight_mat(weight_dict):
    cancat_weight_dict = {}
    for layer_name in weight_dict.keys():
        if layer_name == 'output':
            continue
        if layer_name not in ('fc1',):
            raise NotImplementedError('Unknown layer: ' + layer_name)
        # This is a prayer, but akin to the RNN philosophy below:
        w = weight_dict[layer_name][u'weights:0']
        b = weight_dict[layer_name][u'biases:0']
        # weights are (1000, 330)
        # biases are (330,)
        # tack biases on the end of weights
        b_2d = tf.expand_dims(b, 0)
        flattened = tf.concat((w, b_2d), 0)
        cancat_weight_dict[layer_name] = flattened
    return cancat_weight_dict

    # Original code:
    cancat_weight_dict = {}
    for layer_name in weight_dict.keys():
        if not 'cell' in layer_name:
            cur_w = weight_dict[layer_name][u'weights:0']
            cur_b = weight_dict[layer_name][u'biases:0']
            cur_w_shape = cur_w.get_shape().as_list()
            new_w_shape_a = 1
            for idx in xrange(len(cur_w_shape)-1):
                new_w_shape_a *= cur_w_shape[idx]
            new_w_shape_b = cur_w_shape[-1]
            cur_w = tf.reshape(cur_w, [new_w_shape_a, new_w_shape_b])
            cur_b = tf.expand_dims(cur_b, 0)
            weight = tf.concat(axis=0, values=[cur_w, cur_b])
            cancat_weight_dict[layer_name] = weight
        else:
            gates_w = weight_dict[layer_name][u'gates'][u'kernel:0']
            gates_b = weight_dict[layer_name][u'gates'][u'bias:0']
            candidate_w = weight_dict[layer_name][u'candidate'][u'kernel:0']
            candidate_b = weight_dict[layer_name][u'candidate'][u'bias:0']

            gates_b = tf.expand_dims(gates_b, 0)
            gates_weight_pre = 	tf.concat(axis=0, values=[gates_w, gates_b])
            gates_weight_0, gates_weight_1 = tf.split(axis=1, num_or_size_splits=2, value=gates_weight_pre)
            gates_weight = tf.concat(axis=0, values=[gates_weight_0, gates_weight_1])

            candidate_b = tf.expand_dims(candidate_b, 0)
            candidate_weight = tf.concat(axis=0, values=[candidate_w, candidate_b])

            weight = tf.concat(axis=0, values=[gates_weight, candidate_weight])
            cancat_weight_dict[layer_name] = weight

    return cancat_weight_dict

# XXX TODO XXX
# The if '3' stuff everywhere is dealing with their fancier network structure
#
# All Convolutional:
#
# acc1 -> acc2 -> acc3 \
#                       --> sen1 -> sen2 -> sen3 ->
# gyr1 -> gyr2 -> gyr3 /
#
# Then some GRU:
#
# -> gru1 -> gru2 ->
#
# And a FC to close
#
# -> fc1

# This function is currently a mystery I haven't thought about, we'll see where
# it gets called and the revisit whether we have to do anything to it.
def merg_ord_mat(cancat_weight_dict, ord_list):
    weight_list = []
    for ord_elem in ord_list:
        if type(ord_elem) == type([]):
            sub_weights = [cancat_weight_dict[sub_ord_elem] for sub_ord_elem in ord_elem]
            if '3' in ord_elem[0]:
                weight = tf.concat(axis=0, values=sub_weights)
            else:
                weight = tf.concat(axis=1, values=sub_weights)
        else:
            weight = cancat_weight_dict[ord_elem] 
        weight_list.append(weight)
    return weight_list

# I _think_ this just linearizes weight matricies to a vector for the RNN
def trans_mat2vec(weight_list, vec_dim):
    vec_list = []
    for idx, weight in enumerate(weight_list):
        weight_shape = weight.get_shape().as_list()
        matrix1 = tf.get_variable("trans_W"+str(idx)+"a", [1, weight_shape[0]], tf.float64,
                tf.truncated_normal_initializer(stddev=np.power(2.0/(1 + weight_shape[0]),0.5)))
        matrix2 = tf.get_variable("trans_W"+str(idx)+"b", [weight_shape[1], vec_dim], tf.float64,
                tf.truncated_normal_initializer(stddev=np.power(2.0/(weight_shape[1] + vec_dim),0.5)))
        vec = tf.squeeze(tf.matmul(tf.matmul(matrix1, weight), matrix2), [0])
        vec_list.append(vec)
    return vec_list

# Without thinking deeply, I think this is responsible for converting the
# DeepIoT meta-network output to the drop probabilities
#
# confirm, drop_prob_dict needs to be { layer_name: probability_vector }
def transback(out_list, weight_list, ord_list):
    print('\n' + '-'*10 + ' ' + 'transback' + ' ' + '-'*60)
    pprint.pprint(out_list)
    pprint.pprint(weight_list)
    pprint.pprint(ord_list)

    drop_prob_dict = {}
    for idx in range(len(out_list)):
        cell_out = out_list[idx]
        weight = weight_list[idx]
        layer_name = ord_list[idx]

        cell_out_shape = cell_out.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        maxtrix = tf.get_variable("transBack_W"+str(idx), [cell_out_shape[1], weight_shape[1]], tf.float64,
                tf.truncated_normal_initializer(stddev=np.power(2.0/(cell_out_shape[1] + weight_shape[1]),0.5)))
        drop_out_prob = tf.nn.sigmoid(tf.matmul(cell_out, maxtrix))

        print('.'*50)
        print(idx)
        print(cell_out_shape)
        print(weight_shape)
        print(maxtrix)
        print(drop_out_prob)

        if type(layer_name) == type([]):
            if '3' in layer_name[0]:
                for sub_layer_name in layer_name:
                    drop_prob_dict[sub_layer_name] = drop_out_prob
            else:
                drop_out_prob0, drop_out_prob1 = tf.split(axis=1, num_or_size_splits=2, value=drop_out_prob)
                drop_prob_dict[layer_name[0]] = drop_out_prob0
                drop_prob_dict[layer_name[1]] = drop_out_prob1
        else:
            drop_prob_dict[layer_name] = drop_out_prob
    print('-'*10 + ' ' + 'transback' + ' ' + '-'*60)
    return drop_prob_dict


# This one's a beast, haven't thought through yet
#
# So, if I understand the structure correctly, this assumes that each dropout
# layer is the same size (`inter_dim`), which when you're doing the fancy
# transformations on the CNNs and GRUs I guess you can just make true
def compressor(nn_vars, inter_dim = 64, reuse=False, name='compressor'):
    with tf.variable_scope(name, reuse=reuse) as scope:
        print(nn_vars)
        print('-'*60)
        org_weight_dict = {}
        for var in nn_vars:
            #if '_BN' in var.name:
            #    continue
            #if not 'deepSense/' in var.name:
            #    continue
            #
            # I don't think there's any reason to distrust the caller, but for
            # the moment
            assert 'dense_single' in var.name

            var_name_list = var.name.split('/')
            if len(var_name_list) == 3:
                if not var_name_list[1] in org_weight_dict.keys():
                    org_weight_dict[var_name_list[1]] = {}
                org_weight_dict[var_name_list[1]][var_name_list[2]] = var
            elif len(var_name_list) == 7:
                if not var_name_list[3] in org_weight_dict.keys():
                    org_weight_dict[var_name_list[3]] = {}
                if not var_name_list[5] in org_weight_dict[var_name_list[3]].keys():
                    org_weight_dict[var_name_list[3]][var_name_list[5]] = {}
                org_weight_dict[var_name_list[3]][var_name_list[5]][var_name_list[6]] = var

        # org_weight_dict becomes a nested dict   layer:{var_name: <var>}   , e.g.
        # {'fc1': {'biases:0': <tf.Variable 'dense_single/fc1/biases:0' shape=(330,) dtype=float64_ref>,
        #          'weights:0': <tf.Variable 'dense_single/fc1/weights:0' shape=(1000, 330) dtype=float64_ref>},
        #  'fc2': {'biases:0': <tf.Variable 'dense_single/fc2/biases:0' shape=(11,) dtype=float64_ref>,
        #          'weights:0': <tf.Variable 'dense_single/fc2/weights:0' shape=(330, 11) dtype=float64_ref>}}

        pprint.pprint(org_weight_dict)

        cancat_weight_dict = concat_weight_mat(org_weight_dict)

        pprint.pprint(cancat_weight_dict)

        # ord_list = [[u'acc_conv1', u'gyro_conv1'], [u'acc_conv2', u'gyro_conv2'], [u'acc_conv3', u'gyro_conv3'],
        # 			u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1', u'output']
        #ord_list = [[u'acc_conv1', u'gyro_conv1'], [u'acc_conv2', u'gyro_conv2'], [u'acc_conv3', u'gyro_conv3'],
        #        u'sensor_conv1', u'sensor_conv2', u'sensor_conv3', u'cell_0', u'cell_1']
        ord_list = [u'fc1',]
        weight_list = merg_ord_mat(cancat_weight_dict, ord_list)

        vec_list = trans_mat2vec(weight_list, inter_dim)
        vec_input = tf.stack(vec_list)
        vec_input = tf.expand_dims(vec_input, 0)

        lstm_cell = tf.contrib.rnn.LSTMCell(inter_dim)
        init_state = lstm_cell.zero_state(1, tf.float64)
        cell_output, final_stateTuple = tf.nn.dynamic_rnn(lstm_cell, vec_input, initial_state=init_state, time_major=False)
        cell_output = tf.squeeze(cell_output, [0])
        cell_output_list = tf.split(axis=0, num_or_size_splits=len(ord_list), value=cell_output)

        drop_prob_dict = transback(cell_output_list, weight_list, ord_list)

    return drop_prob_dict

# The return of this method is given to the optimizer for the compressor network
#
# XXX It almost certainly needs to be adapted.
def gen_compressor_loss(
        drop_prob_dict, out_binary_mask,
        batch_size, batchLoss, ema, lossMean, lossStd,
        prune_threshold, compress_done):
    print('\n' + '-'*10 + ' ' + 'gen_compressor_loss' + ' ' + '-'*60)
    pprint.pprint(drop_prob_dict)
    pprint.pprint(out_binary_mask)
    pprint.pprint(batch_size)
    pprint.pprint(batchLoss)
    pprint.pprint(ema)
    pprint.pprint(lossMean)
    pprint.pprint(lossStd)
    pprint.pprint(prune_threshold)
    pprint.pprint(compress_done)

    compsBatchLoss = 0
    for layer_name in drop_prob_dict.keys():
        # Recover the probabilities input into Bernoulli from eq (10)
        drop_prob = DeepIoT_utilities.dropout_prune(drop_prob_dict[layer_name], prune_threshold, compress_done)
        # The actual binary mask, z from eq (10)
        out_binary = out_binary_mask[layer_name]

        # Convolutional layer detail
        if 'conv' in layer_name:
            out_binary = tf.squeeze(out_binary)

        drop_prob  = tf.tile(drop_prob, [batch_size, 1])
        print(drop_prob)
        neg_drop_prob = 1.0 - drop_prob
        neg_out_binary = tf.abs(1.0 - out_binary)
        compsBatchLoss += tf.reduce_sum(tf.log(drop_prob*out_binary + neg_drop_prob*neg_out_binary), 1)

    # Need to add this, maybe TF used to let (50,1292) * (50,) happen, but now needs to be (50,1)
    compsBatchLoss = tf.expand_dims(compsBatchLoss, 1)
    compsBatchLoss *= (batchLoss - ema.average(lossMean))/tf.maximum(tf.cast(1.0, tf.float64), ema.average(lossStd))
    compsLoss = tf.reduce_mean(compsBatchLoss)
    print('-'*10 + ' ' + 'gen_compressor_loss' + ' ' + '-'*60)
    return compsLoss

# XXX This could be less silly, but hey, it'll work
#
# Create a { layer: <tf op> } dictionary of operations to run to update the TF
# variables that are holding the drop probabilities.
def update_drop_op(drop_prob_dict, layer_name_to_probability_variables):
    update_drop_op_dict = {}
    for layer_name in drop_prob_dict.keys():
        prob_list = layer_name_to_probability_variables[layer_name]
        prob_list_shape = prob_list.get_shape().as_list()
        drop_prob = drop_prob_dict[layer_name]
        update_drop_op_dict[layer_name] = tf.assign(prob_list, tf.reshape(drop_prob, prob_list_shape))
    return update_drop_op_dict
