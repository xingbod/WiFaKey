from __future__ import print_function, division
import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import datetime
import Encode
import Channel
import Modulation
import time
import tqdm
import math


def buildMSNet(batch_size=500,iters_max=25,learning=True):
    # get the base graph and generator matrix
    N = 52
    m = 42
    Z=16
    code_n = N
    code_k = N - m
    proto_ldpc = Encode.Proto_LDPC(N, m, Z)
    channel = Channel.BSC(N*Z)
    Z_array = np.array([16, 3, 10, 6])
    code_rate = 1.0 * (N - m) / (N-2)

    Ldpc_PCM = proto_ldpc.Ldpc_PCM
    code_PCM = Ldpc_PCM[0].copy()
    for i in range(0, code_PCM.shape[0]):
        for j in range(0, code_PCM.shape[1]):
            if (code_PCM[i, j] == -1):
                code_PCM[i, j] = 0
            else:
                code_PCM[i, j] = 1

    # network hyper-parameters
    sum_edge_c = np.sum(code_PCM, axis=1)
    sum_edge_v = np.sum(code_PCM, axis=0)
    sum_edge = np.sum(sum_edge_v)
    neurons_per_even_layer = neurons_per_odd_layer = np.sum(sum_edge_v)
    input_output_layer_size = N
    clip_tanh = 10.0

    # train settings
    learning_rate = 0
    train_on_zero_word = False
    numOfWordSim_train = 500 # fix for LFW and all 
    # batch_size = numOfWordSim_train


    ############################     init the connecting matrix between network layers   #################################
    Lift_Matrix1 = []
    Lift_Matrix2 = []
    W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
    W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_output = np.zeros((sum_edge, N), dtype=np.float32)

    # init lifting matrix for cyclic shift
    for t in range(0, 4, 1):
        Lift_M1 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
        Lift_M2 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
        code_PCM1 = Ldpc_PCM[t]
        k = 0
        for j in range(0, code_PCM1.shape[1]):
            for i in range(0, code_PCM1.shape[0]):
                if (code_PCM1[i, j] != -1):
                    Lift_num = code_PCM1[i, j] % Z_array[t]
                    for h in range(0, Z_array[t], 1):
                        Lift_M1[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                    k = k + 1
        k = 0
        for i in range(0, code_PCM1.shape[0]):
            for j in range(0, code_PCM1.shape[1]):
                if (code_PCM1[i, j] != -1):
                    Lift_num = code_PCM1[i, j] % Z_array[t]
                    for h in range(0, Z_array[t], 1):
                        Lift_M2[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                    k = k + 1
        Lift_Matrix1.append(Lift_M1)
        Lift_Matrix2.append(Lift_M2)

    # init W_odd2even  variable node updating
    k = 0
    vec_tmp = np.zeros((sum_edge), dtype=np.float32)  # even layer index read with column
    for j in range(0, code_PCM.shape[1], 1):  # run over the columns
        for i in range(0, code_PCM.shape[0], 1):  # break after the first one
            if (code_PCM[i, j] == 1):  # finding the first one is ok
                num_of_conn = int(np.sum(code_PCM[:, j]))  # get the number of connection of the variable node
                idx = np.argwhere(code_PCM[:, j] == 1)  # get the indexes
                for l in range(0, num_of_conn, 1):  # adding num_of_conn columns to W
                    vec_tmp = np.zeros((sum_edge), dtype=np.float32)
                    for r in range(0, code_PCM.shape[0], 1):  # adding one to the right place
                        if (code_PCM[r, j] == 1 and idx[l][0] != r):
                            idx_row = np.cumsum(code_PCM[r, 0:j + 1])[-1] - 1
                            odd_layer_node_count = 0
                            if r > 0:
                                odd_layer_node_count = np.cumsum(sum_edge_c[0:r])[-1]
                            vec_tmp[idx_row + odd_layer_node_count] = 1  # offset index adding
                    W_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break

    # init W_even2odd  parity check node updating
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
                odd_layer_node_count_1 = 0
                odd_layer_node_count_2 = np.cumsum(sum_edge_c[0:i + 1])[-1]
                if i > 0:
                    odd_layer_node_count_1 = np.cumsum(sum_edge_c[0:i])[-1]
                W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                k += 1  # k is counted in column direction

    # init W_output odd to output
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
                odd_layer_node_count = 0
                if i > 0:
                    odd_layer_node_count = np.cumsum(sum_edge_c[0:i])[-1]
                W_output[odd_layer_node_count + idx_row, k] = 1.0
        k += 1

    # init W_skipconn2even  channel input
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                W_skipconn2even[j, k] = 1.0
                k += 1


    ##############################  bulid four neural networks(Z = 16,3, 10, 6) ############################
    net_dict = {}
    # init the learnable network parameters
    Weights_Var = 0.8 * np.ones(sum_edge, dtype=np.float32)
    Biases_Var = np.zeros(sum_edge, dtype=np.float32) + 0.15
    for i in range(0, iters_max, 1):
        net_dict["Weights_Var{0}".format(i)] = tf.Variable(Weights_Var.copy(), name="Weights_Var".format(i))
        net_dict["Biases_Var{0}".format(i)] = tf.Variable(Biases_Var.copy(), name="Biases_Var".format(i))

    if learning:
        for i in range(0, 25, 1):
            w = np.loadtxt('./LDPC_MetaData/Weights_Var_MS/Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)
            b = np.loadtxt('./LDPC_MetaData/Biases_Var_MS/Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)
            net_dict["Weights_Var{0}".format(i)] = tf.Variable(w.copy(), name="Weights_Var".format(i))
            net_dict["Biases_Var{0}".format(i)] = tf.Variable(b.copy(), name="Biases_Var".format(i))
 
    # the decoding neural network of Z=16
    Z = 16
    xa = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xa')
    ya = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='ya')
    xa_input = tf.transpose(xa, [0, 2, 1])
    net_dict["LLRa{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        #variable node update
        x0 = tf.matmul(xa_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRa{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[0].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        #check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_abs = tf.add(tf.abs(x2_1), 10000 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
        x3 = tf.reduce_min(x2_abs, axis=3)
        x2_2 = -x2_1
        x4 = tf.add(tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)), 1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[0])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRa{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0)) # update the LLR
        # output
        y_output_2 = tf.matmul(net_dict["LLRa{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xa, y_output_3)
        net_dict["ya_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='ya_output'.format(i))
        # calculate loss
        net_dict["lossa{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ya,
                                                                logits=net_dict["ya_output{0}".format(i)]), name='lossa'.format(i))
        # AdamOptimizer
        net_dict["train_stepa{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                    learning_rate).minimize(net_dict["lossa{0}".format(i)],
                                    var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=3
    Z = 3
    xb = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xb')
    yb = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yb')
    xb_input = tf.transpose(xb, [0, 2, 1])
    net_dict["LLRb{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        #variable node update
        x0 = tf.matmul(xb_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRb{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[1].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_abs = tf.add(tf.abs(x2_1), 100 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
        x3 = tf.reduce_min(x2_abs, axis=3)
        x2_2 = -x2_1
        x4 = tf.add(tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)), 1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0), net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRb{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRb{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xb, y_output_3)
        net_dict["yb_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yb_output'.format(i))
        # calculate loss
        net_dict["lossb{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yb,
                                                                logits=net_dict["yb_output{0}".format(i)]), name='lossb'.format(i))
        # AdamOptimizer
        net_dict["train_stepb{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                    learning_rate).minimize(net_dict["lossb{0}".format(i)],
                                    var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=10
    Z = 10
    xc = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xc')
    yc = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yc')
    xc_input = tf.transpose(xc, [0, 2, 1])
    net_dict["LLRc{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        # variable node update
        x0 = tf.matmul(xc_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRc{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[2].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_abs = tf.add(tf.abs(x2_1), 100 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
        x3 = tf.reduce_min(x2_abs, axis=3)
        x2_2 = -x2_1
        x4 = tf.add(tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)), 1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[2])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRc{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRc{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xc, y_output_3)
        net_dict["yc_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yc_output'.format(i))
        # calculate loss
        net_dict["lossc{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yc,
                                                                 logits=net_dict["yc_output{0}".format(i)]), name='lossc'.format(i))
        # AdamOptimizer
        net_dict["train_stepc{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                     learning_rate).minimize(net_dict["lossc{0}".format(i)],
                                     var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=6
    Z = 6
    xd = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xd')
    yd = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yd')
    xd_input = tf.transpose(xd, [0, 2, 1])
    net_dict["LLRd{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        # variable node update
        x0 = tf.matmul(xd_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRd{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[3].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_abs = tf.add(tf.abs(x2_1), 100 * (1 - tf.to_float(tf.abs(x2_1) > 0)))
        x3 = tf.reduce_min(x2_abs, axis=3)
        x2_2 = -x2_1
        x4 = tf.add(tf.zeros((batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer)),
                    1 - 2 * tf.to_float(x2_2 < 0))
        x4_prod = -tf.reduce_prod(x4, axis=3)
        x_output_0 = tf.multiply(x3, tf.sign(x4_prod))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[3])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0), net_dict["Weights_Var{0}".format(i)]),
                            net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRd{0}".format(i + 1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRd{0}".format(i + 1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xd, y_output_3)
        net_dict["yd_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yd_output'.format(i))
        # calculate loss
        net_dict["lossd{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yd,
                                                                                logits=net_dict["yd_output{0}".format(i)]),
                                                                                 name='lossd'.format(i))
        # AdamOptimizer
        net_dict["train_stepd{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                      learning_rate).minimize(net_dict["lossd{0}".format(i)],
                                                                      var_list=[net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])
    return net_dict,xa,ya,xb,yb,xc,yc,xd,yd





def buildSPNet(batch_size=500,iters_max=25,learning=True):
    # get the base graph and generator matrix
    N = 52
    m = 42
    Z=16
    code_n = N
    code_k = N - m
    proto_ldpc = Encode.Proto_LDPC(N, m, Z)
    channel = Channel.BSC(N*Z)
    Z_array = np.array([16, 3, 10, 6])
    code_rate = 1.0 * (N - m) / (N-2)

    Ldpc_PCM = proto_ldpc.Ldpc_PCM
    code_PCM = Ldpc_PCM[0].copy()
    for i in range(0, code_PCM.shape[0]):
        for j in range(0, code_PCM.shape[1]):
            if (code_PCM[i, j] == -1):
                code_PCM[i, j] = 0
            else:
                code_PCM[i, j] = 1

    # network hyper-parameters
    sum_edge_c = np.sum(code_PCM, axis=1)
    sum_edge_v = np.sum(code_PCM, axis=0)
    sum_edge = np.sum(sum_edge_v)
    neurons_per_even_layer = neurons_per_odd_layer = np.sum(sum_edge_v)
    input_output_layer_size = N
    clip_tanh = 10.0

    # train settings
    learning_rate = 0
    train_on_zero_word = False
    numOfWordSim_train = 500 # fix for LFW and all 


       ############################     init the connecting matrix between network layers   #################################
    Lift_Matrix1 = []
    Lift_Matrix2 = []
    W_odd2even = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_skipconn2even = np.zeros((N, sum_edge), dtype=np.float32)
    W_even2odd = np.zeros((sum_edge, sum_edge), dtype=np.float32)
    W_output = np.zeros((sum_edge, N), dtype=np.float32)

    # init lifting matrix for cyclic shift
    for t in range(0, 4, 1):
        Lift_M1 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
        Lift_M2 = np.zeros((neurons_per_odd_layer * Z_array[t], neurons_per_odd_layer * Z_array[t]), np.float32)
        code_PCM1 = Ldpc_PCM[t]
        k = 0
        for j in range(0, code_PCM1.shape[1]):
            for i in range(0, code_PCM1.shape[0]):
                if (code_PCM1[i, j] != -1):
                    Lift_num = code_PCM1[i, j] % Z_array[t]
                    for h in range(0, Z_array[t], 1):
                        Lift_M1[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                    k = k + 1
        k = 0
        for i in range(0, code_PCM1.shape[0]):
            for j in range(0, code_PCM1.shape[1]):
                if (code_PCM1[i, j] != -1):
                    Lift_num = code_PCM1[i, j] % Z_array[t]
                    for h in range(0, Z_array[t], 1):
                        Lift_M2[k * Z_array[t] + h, k * Z_array[t] + (h + Lift_num) % Z_array[t]] = 1
                    k = k + 1
        Lift_Matrix1.append(Lift_M1)
        Lift_Matrix2.append(Lift_M2)

    # init W_odd2even  variable node updating
    k = 0
    vec_tmp = np.zeros((sum_edge), dtype=np.float32)  # even layer index read with column
    for j in range(0, code_PCM.shape[1], 1):  # run over the columns
        for i in range(0, code_PCM.shape[0], 1):  # break after the first one
            if (code_PCM[i, j] == 1):  # finding the first one is ok
                num_of_conn = int(np.sum(code_PCM[:, j]))  # get the number of connection of the variable node
                idx = np.argwhere(code_PCM[:, j] == 1)  # get the indexes
                for l in range(0, num_of_conn, 1):  # adding num_of_conn columns to W
                    vec_tmp = np.zeros((sum_edge), dtype=np.float32)
                    for r in range(0, code_PCM.shape[0], 1):  # adding one to the right place
                        if (code_PCM[r, j] == 1 and idx[l][0] != r):
                            idx_row = np.cumsum(code_PCM[r, 0:j + 1])[-1] - 1
                            odd_layer_node_count = 0
                            if r > 0:
                                odd_layer_node_count = np.cumsum(sum_edge_c[0:r])[-1]
                            vec_tmp[idx_row + odd_layer_node_count] = 1  # offset index adding
                    W_odd2even[:, k] = vec_tmp.transpose()
                    k += 1
                break

    # init W_even2odd  parity check node updating
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
                odd_layer_node_count_1 = 0
                odd_layer_node_count_2 = np.cumsum(sum_edge_c[0:i + 1])[-1]
                if i > 0:
                    odd_layer_node_count_1 = np.cumsum(sum_edge_c[0:i])[-1]
                W_even2odd[k, odd_layer_node_count_1:odd_layer_node_count_2] = 1.0
                W_even2odd[k, odd_layer_node_count_1 + idx_row] = 0.0
                k += 1  # k is counted in column direction

    # init W_output odd to output
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                idx_row = np.cumsum(code_PCM[i, 0:j + 1])[-1] - 1
                idx_col = np.cumsum(code_PCM[0: i + 1, j])[-1] - 1
                odd_layer_node_count = 0
                if i > 0:
                    odd_layer_node_count = np.cumsum(sum_edge_c[0:i])[-1]
                W_output[odd_layer_node_count + idx_row, k] = 1.0
        k += 1

    # init W_skipconn2even  channel input
    k = 0
    for j in range(0, code_PCM.shape[1], 1):
        for i in range(0, code_PCM.shape[0], 1):
            if (code_PCM[i, j] == 1):
                W_skipconn2even[j, k] = 1.0
                k += 1


    ##############################  bulid four neural networks(Z = 16,3, 10, 6) ############################
    net_dict = {}
    # init the learnable network parameters
    Weights_Var = np.ones(sum_edge, dtype=np.float32)
    Biases_Var = np.zeros(sum_edge, dtype=np.float32)
    for i in range(0, iters_max, 1):
        net_dict["Weights_Var{0}".format(i)] = tf.Variable(Weights_Var.copy(), name="Weights_Var".format(i))
        net_dict["Biases_Var{0}".format(i)] = tf.Variable(Biases_Var.copy(), name="Biases_Var".format(i))
 
    # the decoding neural network of Z=16
    Z = 16
    xa = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xa')
    ya = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='ya')
    xa_input = tf.transpose(xa, [0, 2, 1])
    net_dict["LLRa{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        #variable node update
        x0 = tf.matmul(xa_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRa{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[0].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        #check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_clip = 0.5 * tf.clip_by_value(x2_1, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        x2_tanh = tf.tanh(-x2_clip)

        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)
        x_output_0 = -tf.log(tf.div(1 + x3, 1 - x3))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[0])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRa{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0)) # update the LLR
        # output
        y_output_2 = tf.matmul(net_dict["LLRa{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xa, y_output_3)
        net_dict["ya_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='ya_output'.format(i))
        # calculate loss
        net_dict["lossa{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ya,
                                                                logits=net_dict["ya_output{0}".format(i)]), name='lossa'.format(i))
        # AdamOptimizer
        net_dict["train_stepa{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                    learning_rate).minimize(net_dict["lossa{0}".format(i)],
                                    var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=3
    Z = 3
    xb = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xb')
    yb = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yb')
    xb_input = tf.transpose(xb, [0, 2, 1])
    net_dict["LLRb{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        #variable node update
        x0 = tf.matmul(xb_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRb{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[1].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_clip = 0.5 * tf.clip_by_value(x2_1, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        x2_tanh = tf.tanh(-x2_clip)
        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)
        x_output_0 = -tf.log(tf.div(1 + x3, 1 - x3))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0), net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRb{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRb{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xb, y_output_3)
        net_dict["yb_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yb_output'.format(i))
        # calculate loss
        net_dict["lossb{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yb,
                                                                logits=net_dict["yb_output{0}".format(i)]), name='lossb'.format(i))
        # AdamOptimizer
        net_dict["train_stepb{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                    learning_rate).minimize(net_dict["lossb{0}".format(i)],
                                    var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=10
    Z = 10
    xc = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xc')
    yc = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yc')
    xc_input = tf.transpose(xc, [0, 2, 1])
    net_dict["LLRc{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        # variable node update
        x0 = tf.matmul(xc_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRc{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[2].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_clip = 0.5 * tf.clip_by_value(x2_1, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        x2_tanh = tf.tanh(-x2_clip)

        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)
        x_output_0 = -tf.log(tf.div(1 + x3, 1 - x3))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[2])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0),net_dict["Weights_Var{0}".format(i)]), net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRc{0}".format(i+1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRc{0}".format(i+1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xc, y_output_3)
        net_dict["yc_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yc_output'.format(i))
        # calculate loss
        net_dict["lossc{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yc,
                                                                 logits=net_dict["yc_output{0}".format(i)]), name='lossc'.format(i))
        # AdamOptimizer
        net_dict["train_stepc{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                     learning_rate).minimize(net_dict["lossc{0}".format(i)],
                                     var_list = [net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])

    # the decoding neural network of Z=6
    Z = 6
    xd = tf.placeholder(tf.float32, shape=[batch_size, N, Z], name='xd')
    yd = tf.placeholder(tf.float32, shape=[batch_size, N * Z], name='yd')
    xd_input = tf.transpose(xd, [0, 2, 1])
    net_dict["LLRd{0}".format(0)] = tf.zeros((batch_size, Z, sum_edge), dtype=tf.float32)
    for i in range(0, iters_max, 1):
        # variable node update
        x0 = tf.matmul(xd_input, W_skipconn2even)
        x1 = tf.matmul(net_dict["LLRd{0}".format(i)], W_odd2even)
        x2 = tf.add(x0, x1)
        x2 = tf.transpose(x2, [0, 2, 1])
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer * Z])
        x2 = tf.matmul(x2, Lift_Matrix1[3].transpose())
        x2 = tf.reshape(x2, [batch_size, neurons_per_odd_layer, Z])
        x2 = tf.transpose(x2, [0, 2, 1])
        x_tile = tf.tile(x2, multiples=[1, 1, neurons_per_odd_layer])
        W_input_reshape = tf.reshape(W_even2odd.transpose(), [-1])
        # check node update
        x_tile_mul = tf.multiply(x_tile, W_input_reshape)
        x2_1 = tf.reshape(x_tile_mul, [batch_size, Z, neurons_per_odd_layer, neurons_per_odd_layer])
        x2_clip = 0.5 * tf.clip_by_value(x2_1, clip_value_min=-clip_tanh, clip_value_max=clip_tanh)
        x2_tanh = tf.tanh(-x2_clip)
        x2_abs = tf.add(x2_tanh, 1 - tf.to_float(tf.abs(x2_tanh) > 0))
        x3 = tf.reduce_prod(x2_abs, reduction_indices=3)
        x_output_0 = -tf.log(tf.div(1 + x3, 1 - x3))
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        x_output_0 = tf.reshape(x_output_0, [batch_size, Z * neurons_per_odd_layer])
        x_output_0 = tf.matmul(x_output_0, Lift_Matrix2[3])
        x_output_0 = tf.reshape(x_output_0, [batch_size, neurons_per_odd_layer, Z])
        x_output_0 = tf.transpose(x_output_0, [0, 2, 1])
        # add learnable parameters
        x_output_1 = tf.add(tf.multiply(tf.abs(x_output_0), net_dict["Weights_Var{0}".format(i)]),
                            net_dict["Biases_Var{0}".format(i)])
        x_output_1 = tf.multiply(x_output_1, tf.to_float(x_output_1 > 0))
        net_dict["LLRd{0}".format(i + 1)] = tf.multiply(x_output_1, tf.sign(x_output_0))
        # output
        y_output_2 = tf.matmul(net_dict["LLRd{0}".format(i + 1)], W_output)
        y_output_3 = tf.transpose(y_output_2, [0, 2, 1])
        y_output_4 = tf.add(xd, y_output_3)
        net_dict["yd_output{0}".format(i)] = tf.reshape(y_output_4, [batch_size, N * Z], name='yd_output'.format(i))
        # calculate loss
        net_dict["lossd{0}".format(i)] = 1.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=yd,
                                                                                logits=net_dict["yd_output{0}".format(i)]),
                                                                                 name='lossd'.format(i))
        # AdamOptimizer
        net_dict["train_stepd{0}".format(i)] = tf.train.AdamOptimizer(learning_rate=
                                                                      learning_rate).minimize(net_dict["lossd{0}".format(i)],
                                                                      var_list=[net_dict["Weights_Var{0}".format(i)], net_dict["Biases_Var{0}".format(i)]])


        return net_dict,xa,ya,xb,yb,xc,yc,xd,yd