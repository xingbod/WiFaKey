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



from util import equal_probable,equal_space,onehot_binary,lssc_binary,brgc_binary,look4noncerate,d_prime

from utils.verification import evaluate,evaluate_binary





# basic args

parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--modelname', type=str,
                    default="embeddings_adaface_ir101_webface12m_lfw",
                    help='The cache folder for validation report')

parser.add_argument('--learning', type=bool, default=False,

                    help='whether learning')

parser.add_argument('--gpu', type=str,default='0',

                    help='gpu')

parser.add_argument('--ds', type=str,

                    default='lfw',

                    help='The evaluation dataset')

args = parser.parse_args()



# get the base graph and generator matrix

code_PCM0 = np.loadtxt("./BaseGraph/BaseGraph2_Set0.txt", int, delimiter='	')

code_PCM1 = np.loadtxt("./BaseGraph/BaseGraph2_Set1.txt", int, delimiter='	')

code_PCM2 = np.loadtxt("./BaseGraph/BaseGraph2_Set2.txt", int, delimiter='	')

code_GM_16 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_16.txt", int, delimiter=',')

code_GM_3 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_3.txt", int, delimiter=',')

code_GM_10 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_10.txt", int, delimiter=',')

code_GM_6 = np.loadtxt("./BaseGraph_GM/LDPC_GM_BG2_6.txt", int, delimiter=',')

code_PCM = code_PCM0.copy()

Ldpc_PCM = [code_PCM0, code_PCM1, code_PCM2, code_PCM1]# four LDPC codes with different code lengths

Ldpc_GM = [code_GM_16, code_GM_3, code_GM_10, code_GM_6]





class Timer(object):

    def __init__(self, name=None):

        self.name = name



    def __enter__(self):

        self.tstart = time.time()



    def __exit__(self, type, value, traceback):

        if self.name:

            print('[%s]' % self.name,)

        print('Elapsed: %s' % (time.time() - self.tstart))

        

# GPU settings

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

config = tf.ConfigProto()

config.gpu_options.allow_growth = True

sess = tf.Session(config=config)



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

iters_max = 25     # number of iterations 25

sum_edge_c = np.sum(code_PCM, axis=1)

sum_edge_v = np.sum(code_PCM, axis=0)

sum_edge = np.sum(sum_edge_v)

neurons_per_even_layer = neurons_per_odd_layer = np.sum(sum_edge_v)

input_output_layer_size = N

clip_tanh = 10.0



# init the AWGN #



# train SNR

SNR_Matrix = np.array([[9.0,6.05,4.1,2.95,2.25,1.8,1.55,1.3,1.15,1.05,0.94,0.85,0.83,0.81,0.8,0.8,0.8,0.75,0.75,0.7,0.7,0.7,0.7,0.7,0.7],

                       [9.1,6.2,4.6,3.7,3.2,3.0,2.8,2.7,2.6,2.55,2.5,2.45,2.4,2.4,2.4,2.35,2.35,2.3,2.3,2.3,2.25,2.25,2.25,2.25,2.25],

                       [9,6.05,4.1,3,2.4,2,1.7,1.5,1.4,1.4,1.3,1.3,1.2,1.2,1.2,1.2,1.1,1.1,1.1,1.1,1.1,1.1,1.1,1,1],

                       [9.0,6.1,4.25,3.2,2.6,2.25,2,1.9,1.8,1.7,1.7,1.65,1.6,1.6,1.55,1.55,1.5,1.5,1.5,1.45,1.45,1.4,1.4,1.4,1.4]])

                       

# crossover_prob_set = np.linspace(0.06,0.06,1,dtype=np.float32)

Cross_over =  np.array([[0.003,0.007020101,0.067221105,0.15150252,0.20568341,0.22976382,0.24481407,0.24782412,0.27190453,0.27190453,0.27190453,0.27190453,0.27190453,0.27491456,0.27491456,0.27491456,0.27491456,0.27792463,0.27792463,0.27792463,0.27792463,0.27792463,0.27792463,0.27792463,0.27792463],

[0.001,0.019121213,0.09764647,0.17013131,0.19731313,0.20637374,0.21241415,0.21845454,0.23959596,0.24563636,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697,0.25469697],

[0.001,0.019121213,0.12784849,0.15503031,0.17315151,0.18825252,0.19429293,0.20033333,0.21543434,0.22751515,0.23959596,0.24563636,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677,0.25167677],

[0.001,0.022141414,0.09764647,0.1248282,0.13388889,0.1459697,0.1520101,0.16711111,0.17013131,0.17013131,0.17315151,0.17315151,0.17315151,0.17315151,0.19429293,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333,0.20033333]])





SNR_lin = 10.0 ** (SNR_Matrix / 10.0)

SNR_sigma = np.sqrt(1.0 / (2.0 * SNR_lin * code_rate))/10.

# SNR_sigma = Cross_over

# ramdom seed

word_seed = 2042

noise_seed = 1074

wordRandom = np.random.RandomState(word_seed)  # word seed

noiseRandom = np.random.RandomState(noise_seed)  # noise seed



# train settings

learning_rate = 0

train_on_zero_word = False

numOfWordSim_train = 500 # fix for LFW and all 

batch_size = numOfWordSim_train

num_of_batch = 1



# modelname = "embeddings_ViT_P12S8"

# modelname = "embeddings0_Magface_ir100_MS1MV2_lfw"

# modelname = "embeddings_ArcFace_ir50"

modelname = args.modelname

learning = args.learning

ds = args.ds



print('modelname:',modelname,'ds:',ds,'learning:',learning)



embeddings_o = np.loadtxt('embeddings/'+modelname+'.csv', delimiter=',') 

issame = np.loadtxt('embeddings/'+ds+'_issame.csv', delimiter=',') > 0





intervals = equal_probable(embeddings_o,intervalnum=4)

# intervals = equal_space(embeddings_o,intervalnum=6)

print(intervals)

embeddings = lssc_binary(embeddings_o,interval = intervals)



###

print("#####orignal acc eucliden######")

tpr, fpr, accuracy, best_thresholds = evaluate(embeddings_o, issame, 10)

d_p, genmean,impmean = d_prime(embeddings_o,issame,dist='eucliden')

print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)

print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

    val_list = ['eucliden', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]

    log = '\t'.join(str(value) for value in val_list)

    f.writelines(log + '\n')

    

print("#####orignal acc cosine######")

tpr, fpr, accuracy, best_thresholds = evaluate(embeddings_o, issame, 10)

d_p, genmean,impmean  = d_prime(embeddings_o,issame,dist='cosine')

print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)

print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

    val_list = ['cosine', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]

    log = '\t'.join(str(value) for value in val_list)

    f.writelines(log + '\n')

    

print("#####orignal sign acc######")

tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_o>0, issame, 10)

d_p, genmean,impmean  = d_prime(embeddings_o>0,issame,dist='hamming')

print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)

print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

    val_list = ['sign', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]

    log = '\t'.join(str(value) for value in val_list)

    f.writelines(log + '\n')

    

print("#####binary acc######")

tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings, issame, 10)

d_p, genmean,impmean  = d_prime(embeddings,issame,dist='hamming')

print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)

print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

    val_list = ['lssc_binary', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]

    log = '\t'.join(str(value) for value in val_list)

    f.writelines(log + '\n')

    

print("#####look4noncerate######")

# look4noncerate(embeddings,issame,threshold = 0.15, genorimp=1,confidence = 0.95)

# embeddings_nonce,crossover_prob,nonce = look4noncerate(embeddings,issame,threshold = 0.15, genorimp=1,confidence = 0.95)# reply on ecc

embeddings_nonce,crossover_prob,nonce = look4noncerate(embeddings,issame,threshold = 0.23, genorimp=0,confidence = 0.95)# 95% imposters scores above 0.25

tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce, issame, 10)

d_p, genmean,impmean  = d_prime(embeddings_nonce,issame,dist='hamming')

print('crossover_prob',crossover_prob,',accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p, np.mean(genmean),np.mean(impmean))

print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

    val_list = ['addnonce', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]

    log = '\t'.join(str(value) for value in val_list)

    f.writelines(log + '\n')





#####################################################################

# in_dim = 512

# out_dim = 832

# np.random.seed(0)# 这里fix seed 固定结果，实验的时候可以注释掉

# proj = np.random.normal(0, 1, (in_dim, out_dim))# 这里是一个矩阵，随机映射矩阵

# def BioHashing(data, proj):

#     data = np.array(data)

#     y = np.matmul(data, proj)

#     return (y>0)*1



# embeddings_o = np.loadtxt('embeddings/embeddings_adaface_ir101_webface12m_lfw.csv', delimiter=',') 

# embeddings = np.zeros((12000,out_dim))

# for i in tqdm.tqdm(range(12000)):

#     # embeddings[i,:] = IndexRankingHashing2(embeddings_o[i,:], proj,in_dim = in_dim, out_dim = out_dim, q = 2)

#     embeddings[i,:] = BioHashing(embeddings_o[i,:],proj)

#######################################################################

# interval = np.array([-1, 0 ,1.0])

# lkut = np.zeros((len(interval)+1,len(interval))) 

# for i in range(1,len(interval)+1):

#     lkut[i,len(interval)-i:] = 1

# print(lkut)

# def LSSC(data):

#     new_data = np.zeros(512*len(interval))

#     for i in range(len(data)):

#         index = np.where(interval>data[i])[0][0]

#         # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])

#         new_data[i*len(interval):(i+1)*len(interval)] = lkut[index]

#     return new_data

# block = len(interval)# 3 

 

# embeddings = np.zeros((12000,512*block))

# for i in tqdm.tqdm(range(len(embeddings))):

#     embeddings[i,:] =  LSSC(embeddings_o[i,:])

########################################################################



# interval = np.array([-0.25, 0 , 0.25])

# interval = np.array([-0.1,-0.05,-0.025,0,0.025, 0.05,0.1])

# lkut = np.zeros((len(interval)+1,len(interval))) 

# for i in range(1,len(interval)+1):

#     lkut[i,len(interval)-i:] = 1

# print(lkut)

# def LSSC(data):

#     new_data = np.zeros(512*len(interval))

#     for i in range(len(data)):

#         # print(interval>data[i],data[i])

#         index = -1

#         whereindex = np.where(interval>data[i])

#         if len(whereindex[0]) != 0:

#             index = whereindex[0][0]

#         # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])

#         new_data[i*len(interval):(i+1)*len(interval)] = lkut[index]

#     return new_data

# block = len(interval) 

 

# embeddings = np.zeros((len(embeddings_o),512*block))

# for i in tqdm.tqdm(range(len(embeddings_o))):

#     embeddings[i,:] =  LSSC(embeddings_o[i,:])





##########################################################

emb1 = embeddings_nonce[0::2]==0

emb2 = embeddings_nonce[1::2]==0



bio_noise = np.logical_xor(emb1,emb2)*1  # same -> 0 



gen = bio_noise[issame==1]

imp = bio_noise[issame==0]



bio_noise = -2*bio_noise+1

                

# gen = np.sum(gen,axis = 1)

# imp = np.sum(imp,axis = 1)



# Dprime= np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))

# print('Dprime',Dprime, 'Hamming Dist, gen: ',(np.mean(gen))/embeddings.shape[1],', imp:', (np.mean(imp))/embeddings.shape[1])



# with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

#     val_list = [Dprime, (np.mean(gen))/embeddings.shape[1], (np.mean(imp))/embeddings.shape[1]]

#     log = '\t'.join(str(value) for value in val_list)

#     f.writelines(log + '\n')

                

# embeddings = np.concatenate([embeddings_o,np.ones((len(embeddings_o),8))],axis =1)  > 0



#get train samples

def create_mix_epoch(crossover_prob, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word):

    X = np.zeros([1, code_n * Z], dtype=np.float32)

    Y = np.zeros([1, code_n * Z], dtype=np.int64)



    # build set for epoch

    for sf_i in crossover_prob:

        if is_zeros_word:

            infoWord_i = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

        else:

            infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

        

        u_coded_bits = proto_ldpc.encode_LDPC(infoWord_i)

        

        u_coded_bits = Modulation.BPSK(u_coded_bits)

        y_receive, ch_noise = channel.channel_transmit( numOfWordSim, u_coded_bits, sf_i)

        

        X = np.vstack((X, y_receive))

        Y = np.vstack((Y, u_coded_bits))

    X = X[1:]

    Y = Y[1:]

    X = np.reshape(X, [batch_size, code_n, Z])

    return X, Y





def create_biometric_batch(crossover_prob, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word, issamefalg = 1, fold = 0):# 500 * 6

    X = np.zeros([1, code_n * Z], dtype=np.float32)

    Y = np.zeros([1, code_n * Z], dtype=np.int64)



    # build set for epoch

    # is_zeros_word = True

    if is_zeros_word:

        infoWord_i = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

    else:

        infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))

        

    # infoWord_i = Modulation.BPSK(infoWord_i)

    u_coded_bits = proto_ldpc.encode_LDPC(infoWord_i)

    s_mod = Modulation.BPSK(u_coded_bits)

    

    # noise_bsc = np.sign(np.random.random(size=s_mod.shape)-crossover_prob)

    

    same_noise = bio_noise[issame==issamefalg,:code_n * Z]

    noise_bsc = same_noise[fold*500:(fold+1)*500,:]

    

    y_receive = np.multiply(s_mod,noise_bsc)



    X = np.vstack((X, y_receive))

    Y = np.vstack((Y, u_coded_bits))

    

    X = X[1:]

    Y = Y[1:]

    X = np.reshape(X, [batch_size, code_n, Z])

    return X, Y



# calculate ber and fer

def calc_ber_fer(snr_db, Y_test_pred, Y_test, numOfWordSim):

    ber_test = np.zeros(snr_db.shape[0])

    fer_test = np.zeros(snr_db.shape[0])

    for i in range(0, snr_db.shape[0]):

        Y_test_pred_i = Y_test_pred[i * numOfWordSim:(i + 1) * numOfWordSim, :]

        Y_test_i = Y_test[i * numOfWordSim:(i + 1) * numOfWordSim, :]

        ber_test[i] = np.abs(((Y_test_pred_i > 0) - Y_test_i)).sum() / (Y_test_i.shape[0] * Y_test_i.shape[1])

        fer_test[i] = (np.abs(((Y_test_pred_i > 0) - Y_test_i)).sum(axis=1) > 0).sum() * 1.0 / Y_test_i.shape[0]

    return ber_test, fer_test





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

# for i in range(0, iters_max, 1):

#     if i<16:

#         w = np.loadtxt('./Weights_Var_BSC/SP_BSC_Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

#         b = np.loadtxt('./Biases_Var_BSC/SP_BSC_Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

#         # w = np.loadtxt('./Weights_Var/SP_BSC_Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

#         # b = np.loadtxt('./Biases_Var/SP_BSC_Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

#         net_dict["Weights_Var{0}".format(i)] = tf.Variable(w.copy(), name="Weights_Var".format(i))

#         net_dict["Biases_Var{0}".format(i)] = tf.Variable(b.copy(), name="Biases_Var".format(i))

#     else:

#         net_dict["Weights_Var{0}".format(i)] = tf.Variable(Weights_Var.copy(), name="Weights_Var".format(i))

#         net_dict["Biases_Var{0}".format(i)] = tf.Variable(Biases_Var.copy(), name="Biases_Var".format(i))

#         # net_dict["Weights_Var{0}".format(i)] = tf.Variable(w.copy(), name="Weights_Var".format(i))

#         # net_dict["Biases_Var{0}".format(i)] = tf.Variable(b.copy(), name="Biases_Var".format(i))



for i in range(0, iters_max, 1):

    net_dict["Weights_Var{0}".format(i)] = tf.Variable(Weights_Var.copy(), name="Weights_Var".format(i))

    net_dict["Biases_Var{0}".format(i)] = tf.Variable(Biases_Var.copy(), name="Biases_Var".format(i))

if learning:

    for i in range(0, 25, 1):

        # w = np.loadtxt('./Weights_Var_BSC_z16/SP_BSC_Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

        # b = np.loadtxt('./Biases_Var_BSC_z16/SP_BSC_Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

        # w = np.loadtxt('./Weights_Var/SP_BSC_Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

        # b = np.loadtxt('./Biases_Var/SP_BSC_Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

        w = np.loadtxt('./Weights_Var_SP/Weights_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

        b = np.loadtxt('./Biases_Var_SP/Biases_Var{0}.txt'.format(i), delimiter=',', dtype=np.float32)

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





##################################  Test  ####################################

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for iter in range(0, iters_max, 1):

    for i in range(0, num_of_batch, 1):

        #ramdom choose Z with different probabilities

        index = [0, 1, 2, 3, 0, 0]

        # Z_type = index[wordRandom.randint(0, 6)]

        Z_type = 0

        if Z_type == 0:

            Z = 16

            # crossover_prob, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word

            # N = 52

            # m = 42

            # Z=16

            # code_n = N

            # code_k = N - m

            proto_ldpc = Encode.Proto_LDPC(N, m, Z)

            channel = Channel.BSC(N*Z)



#             training_received_data, training_coded_bits = create_mix_epoch(SNR_set, wordRandom, noiseRandom, numOfWordSim_train,

#                                                                            code_n, code_k, channel,Z,

#                                                                            proto_ldpc,

#                                                                            train_on_zero_word)

#             training_labels_for_mse = training_coded_bits

#             y_pred, train_loss = sess.run(fetches=[net_dict["ya_output{0}".format(iter)], net_dict["lossa{0}".format(iter)]],

#                                              feed_dict={xa: training_received_data, ya: training_labels_for_mse})



#             hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

#             print('A iteration: [{0}/{1}]\t'

#                   'epoch: [{2}/{3}]\t'

#                   'loss: {4}\t'

#                   'mean matched bits: {5}\t'

#                   'full rec.: {6}\t '

#                   'Cross: {7}\t'.format(

#                 iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hits),np.sum(hits>0.99999),SNR_set[0]))

            

            #####################For Bio data ###############################

            SNR_set = [0.1]

            totalfold = 6

            if ds == 'cfp_fp' or ds =='cfp_ff':

                totalfold = 7

            t = time.time()

            hitss = []

            for fold in range(totalfold):

                training_received_data, training_coded_bits = create_biometric_batch(SNR_set, wordRandom, noiseRandom,                                                                                      numOfWordSim_train,

                                                                               code_n, code_k, channel,Z,

                                                                               proto_ldpc,

                                                                               train_on_zero_word, issamefalg = 1, fold = fold)

                training_labels_for_mse = training_coded_bits

                y_pred, train_loss = sess.run(fetches=[net_dict["ya_output{0}".format(iter)], net_dict["lossa{0}".format(iter)]],

                                                 feed_dict={xa: training_received_data, ya: training_labels_for_mse})    

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                hitss.append(hits)

            hitss =np.array(hitss)

            print('B iteration: [{0}/{1}]\t'

                  'epoch: [{2}/{3}]\t'

                  'loss: {4}\t'

                  'mean matched bits: {5}\t'

                  'full rec.: {6}\t'.format(

                iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hitss),np.sum(hitss>0.99999)))

            elapsed = time.time() - t



            # t = time.time()

            hitss2 = []

            for fold in range(totalfold):

                training_received_data, training_coded_bits = create_biometric_batch(SNR_set, wordRandom, noiseRandom,                                                                                      numOfWordSim_train,

                                                                               code_n, code_k, channel,Z,

                                                                               proto_ldpc,

                                                                               train_on_zero_word, issamefalg = 0, fold = fold)

                training_labels_for_mse = training_coded_bits

                y_pred, train_loss2, _ = sess.run(fetches=[net_dict["ya_output{0}".format(iter)], net_dict["lossa{0}".format(iter)],

                                                          net_dict["train_stepa{0}".format(iter)]],

                                                 feed_dict={xa: training_received_data, ya: training_labels_for_mse})    

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                hitss2.append(hits)

            hitss2 =np.array(hitss2)

            print('C iteration: [{0}/{1}]\t'

                  'epoch: [{2}/{3}]\t'

                  'loss: {4}\t'

                  'mean matched bits: {5}\t'

                  'full rec.: {6}\t'.format(

                iter + 1, iters_max, i, num_of_batch, train_loss2,np.mean(hitss2),np.sum(hitss2>0.99999)))

            elapsed = time.time() - t

            

            with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

                val_list = [iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hitss),np.sum(hitss>0.999999999999), train_loss2,np.mean(hitss2),np.sum(hitss2>0.999999999999),elapsed]

                log = '\t'.join(str(value) for value in val_list)

                f.writelines(log + '\n')



        elif Z_type == 1:

            Z = 3

            SNR_set = np.array([SNR_sigma[Z_type, iter]])

            proto_ldpc = Encode.Proto_LDPC(N, m, Z)

            channel = Channel.BSC(N*Z)

            crossover_prob_set = np.linspace(0.06,0.06,1,dtype=np.float32)



            training_received_data, training_coded_bits = create_mix_epoch(SNR_set, wordRandom, noiseRandom, numOfWordSim_train,

                                                                           code_n, code_k, channel,Z,

                                                                           proto_ldpc,

                                                                           train_on_zero_word)



            training_labels_for_mse = training_coded_bits

            y_pred, train_loss, _ = sess.run(fetches=[net_dict["yb_output{0}".format(iter)], net_dict["lossb{0}".format(iter)],

                                                      net_dict["train_stepb{0}".format(iter)]],

                                             feed_dict={xb: training_received_data, yb: training_labels_for_mse})

            if i % 200 == 0:

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                print('iteration: [{0}/{1}]\t'

                      'epoch: [{2}/{3}]\t'

                      'loss: {4}\t'

                      'mean matched bits: {5}\t'

                      'full rec.: {6}\t'.format(

                    iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hits),np.sum(hits>0.99999)))

                



        elif Z_type == 2:

            Z = 10

            proto_ldpc = Encode.Proto_LDPC(N, m, Z)

            channel = Channel.BSC(N*Z)

            crossover_prob_set = np.linspace(0.06,0.06,1,dtype=np.float32)

            SNR_set = np.array([SNR_sigma[Z_type, 1]])



#             training_received_data, training_coded_bits = create_mix_epoch(SNR_set, wordRandom, noiseRandom, numOfWordSim_train,

#                                                                            code_n, code_k, channel,Z,

#                                                                            proto_ldpc,

#                                                                            train_on_zero_word)



#             training_labels_for_mse = training_coded_bits

#             y_pred, train_loss, _ = sess.run(fetches=[net_dict["yc_output{0}".format(iter)], net_dict["lossc{0}".format(iter)],

#                                                        net_dict["train_stepc{0}".format(iter)]],

#                                               feed_dict={xc: training_received_data, yc: training_labels_for_mse})



#             hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

#             print('A iteration: [{0}/{1}]\t'

#                   'epoch: [{2}/{3}]\t'

#                   'loss: {4}\t'

#                   'mean matched bits: {5}\t'

#                   'full rec.: {6}\t '

#                   'Cross: {7}\t'.format(

#                 iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hits),np.sum(hits>0.99999),SNR_set[0]))

            

            #####################For Bio data ###############################

            # with Timer('foo_stuff'):

            t = time.time()

            hitss = []

            totalfold = 6

            if ds == 'cfp_fp' or ds =='cfp_ff':

                totalfold = 7

            for fold in range(totalfold):

                training_received_data, training_coded_bits = create_biometric_batch(SNR_set, wordRandom, noiseRandom,                                                                                      numOfWordSim_train,

                                                                               code_n, code_k, channel,Z,

                                                                               proto_ldpc,

                                                                               train_on_zero_word, issamefalg = 1, fold = fold)

                training_labels_for_mse = training_coded_bits

                y_pred, train_loss = sess.run(fetches=[net_dict["yc_output{0}".format(iter)], net_dict["lossc{0}".format(iter)]],

                                                 feed_dict={xc: training_received_data, yc: training_labels_for_mse})    

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                hitss.append(hits)

            hitss =np.array(hitss)

            elapsed = time.time() - t

            print('B iteration: [{0}/{1}]\t'

                  'epoch: [{2}/{3}]\t'

                  'loss: {4}\t'

                  'mean matched bits: {5}\t'

                  'full rec.: {6}, {7}\t'.format(

                iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hitss),np.sum(hitss>0.99999),elapsed))

            

            t = time.time()

            hitss2 = []

            for fold in range(6):

                training_received_data, training_coded_bits = create_biometric_batch(SNR_set, wordRandom, noiseRandom,                                                                                      numOfWordSim_train,

                                                                               code_n, code_k, channel,Z,

                                                                               proto_ldpc,

                                                                               train_on_zero_word, issamefalg = 0, fold = fold)

                training_labels_for_mse = training_coded_bits

                y_pred, train_loss2 = sess.run(fetches=[net_dict["yc_output{0}".format(iter)], net_dict["lossc{0}".format(iter)]],

                                                 feed_dict={xc: training_received_data, yc: training_labels_for_mse})    

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                hitss2.append(hits)

            hitss2 =np.array(hitss2)

            print('C iteration: [{0}/{1}]\t'

                  'epoch: [{2}/{3}]\t'

                  'loss: {4}\t'

                  'mean matched bits: {5}\t'

                  'full rec.: {6}, {7}\t'.format(

                iter + 1, iters_max, i, num_of_batch, train_loss2,np.mean(hitss2),np.sum(hitss2>0.99999),elapsed))

            with open('logs/log_sp_bsc_'+modelname+'_'+str(learning)+'.txt', 'a') as f:

                val_list = [iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hitss),np.sum(hitss>0.999999999999), train_loss2,np.mean(hitss2),np.sum(hitss2>0.999999999999),elapsed]

                log = '\t'.join(str(value) for value in val_list)

                f.writelines(log + '\n')



        

        else:

            Z = 6

            SNR_set = np.array([SNR_sigma[Z_type, iter]])

            # (crossover_prob, wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word)

            proto_ldpc = Encode.Proto_LDPC(N, m, Z)

            channel = Channel.BSC(N*Z)

            crossover_prob_set = np.linspace(0.06,0.06,1,dtype=np.float32)



            training_received_data, training_coded_bits = create_mix_epoch(SNR_set, wordRandom, noiseRandom, numOfWordSim_train,

                                                                           code_n, code_k, channel,Z,

                                                                           proto_ldpc,

                                                                           train_on_zero_word)



            training_labels_for_mse = training_coded_bits

            y_pred, train_loss, _ = sess.run(fetches=[net_dict["yd_output{0}".format(iter)], net_dict["lossd{0}".format(iter)],

                                                       net_dict["train_stepd{0}".format(iter)]],

                                              feed_dict={xd: training_received_data, yd: training_labels_for_mse})

            if i % 200 == 0:

                hits = np.sum(((y_pred>0)*1==training_labels_for_mse)*1,axis=1)/len(y_pred[0])

                print('iteration: [{0}/{1}]\t'

                      'epoch: [{2}/{3}]\t'

                      'loss: {4}\t'

                      'mean matched bits: {5}\t'

                      'full rec.: {6}\t'.format(

                    iter + 1, iters_max, i, num_of_batch, train_loss,np.mean(hits),np.sum(hits>0.99999)))

                



