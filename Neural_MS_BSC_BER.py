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

from util import equal_probable,equal_space,onehot_binary,lssc_binary,brgc_binary,look4noncerate,d_prime,adduserspecfkey
from utils.verification import evaluate,evaluate_binary
from NeuralUtils import buildMSNet, buildSPNet


# basic args
parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--learning', type=bool, default=False,
                    help='whether learning')
parser.add_argument('--gpu', type=str,default='0',
                    help='gpu')
parser.add_argument('--decoder', type=str,default='ms',
                    help='decoder: ms sp')
parser.add_argument('--z', type=int,
                    default=10,
                    help='Z factor')
parser.add_argument('--startiter', type=int,
                    default=0,
                    help='start iter number, default 0, 0-24')
parser.add_argument('--iters_max', type=int,
                    default=25,
                    help='iters_max')

args = parser.parse_args()

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

crossover_prob_set = np.linspace(0.001,0.3,100,dtype=np.float32)


# network hyper-parameters
iters_max = args.iters_max     # number of iterations 25
N = 52
m = 42
code_n = N
code_k = N - m

# ramdom seed
word_seed = 2042
noise_seed = 1074
wordRandom = np.random.RandomState(word_seed)  # word seed
noiseRandom = np.random.RandomState(noise_seed)  # noise seed

# train settings
train_on_zero_word = False
batch_size = 500 # fix for LFW and all 


learning = args.learning


#get train samples
def create_mix_epoch(crossover_prob, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word):
    X = np.zeros([1, code_n * Z], dtype=np.float32)
    Y = np.zeros([1, code_n * Z], dtype=np.int64)

    # build set for epoch
    if is_zeros_word:
        infoWord_i = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))
    else:
        infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

    u_coded_bits = proto_ldpc.encode_LDPC(infoWord_i)
    s_mod = Modulation.BPSK(u_coded_bits)
    
    noise_bsc = np.sign(np.random.random(size=s_mod.shape)-crossover_prob)
    
    
    y_receive = np.multiply(s_mod,noise_bsc)


    X = np.vstack((X, y_receive))
    Y = np.vstack((Y, u_coded_bits))
    X = X[1:]
    Y = Y[1:]
    X = np.reshape(X, [batch_size, code_n, Z])
    return X, Y



#### learning max 25
print(args.decoder)
if args.decoder == 'ms':
    net_dict,xa,ya,xb,yb,xc,yc,xd,yd = buildMSNet(batch_size=batch_size,iters_max=args.iters_max,learning=args.learning)
elif args.decoder == 'sp':
    net_dict,xa,ya,xb,yb,xc,yc,xd,yd = buildSPNet(batch_size=batch_size,iters_max=args.iters_max,learning=False)
else:
    print('no decoder choose!')
# Z = 10

if args.z == 16:
    Z = args.z
elif args.z == 10:
    Z = args.z

proto_ldpc = Encode.Proto_LDPC(N, m, Z)
channel = Channel.BSC(N*Z)


##################################  Test  ####################################
sess.run(tf.global_variables_initializer())

iterss = [i for i in range(25)]
fetchkeysa = []
fetchkeysb = []
fetchkeysc = []
fetchkeysd = []
for i in iterss:
    fetchkeysa.append(net_dict["ya_output{0}".format(i)])
    fetchkeysb.append(net_dict["yb_output{0}".format(i)])
    fetchkeysc.append(net_dict["yc_output{0}".format(i)])
    # fetchkeysd.append(net_dict["yd_output{0}".format(i)])


#####################For Bio data ###############################

totalfold = 10
t = time.time()
hitss = np.zeros((len(iterss),totalfold))
hitss_imposter = np.zeros((len(iterss),totalfold))
block_nums = 1 
for crossover_i in crossover_prob_set:
    for fold in range(totalfold):
        GT = np.zeros([batch_size, code_n * Z * block_nums], dtype=np.float32)
        Pred = np.zeros([len(iterss), batch_size, code_n * Z * block_nums], dtype=np.float32) # pay attention
        for blockindex in range(block_nums): # bio data is split into blocks 
            training_received_data, training_coded_bits = create_mix_epoch(crossover_i, wordRandom, noiseRandom, batch_size,
                                                                       code_n, code_k, channel,Z,
                                                                       proto_ldpc,
                                                                       train_on_zero_word)
            training_labels_for_mse = training_coded_bits
            if Z==10:
                y_pred = sess.run(fetches=fetchkeysc, feed_dict={xc: training_received_data, yc: training_labels_for_mse})   
            elif Z== 16:
                y_pred = sess.run(fetches=fetchkeysa, feed_dict={xa: training_received_data, ya: training_labels_for_mse}) 
            elif Z== 6:
                y_pred = sess.run(fetches=fetchkeysd, feed_dict={xd: training_received_data, yd: training_labels_for_mse}) 
            elif Z== 3:
                y_pred = sess.run(fetches=fetchkeysb, feed_dict={xb: training_received_data, yb: training_labels_for_mse}) 

            GT[:,blockindex* code_n * Z:(blockindex+1)* code_n * Z] = training_coded_bits
            Pred[:,:,blockindex* code_n * Z:(blockindex+1)* code_n * Z] = y_pred # (33,500,520)

        # here cal all iter at once to save time
        for ttt in range(len(iterss)):
            hit = np.sum(((Pred[ttt,:,:]>0)*1==GT)*1,axis=1)/(code_n * Z * block_nums)
            hitss[ttt,fold] = np.sum(hit>0.9999999999999)

    gen_hits = np.sum(hitss,axis=1)

    with open('logs/log_ms_bsc_BER'+args.decoder+str(learning)+'.txt', 'a') as f:
        log = str(crossover_i) + '\t' + '\t'.join(str(value) for value in gen_hits)
        f.writelines(log + '\n')