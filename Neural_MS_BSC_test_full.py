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
parser.add_argument('--modelname', type=str,
                    help='The cache folder for validation report')
parser.add_argument('--learning', type=bool, default=False,
                    help='whether learning')
parser.add_argument('--gpu', type=str,default='0',
                    help='gpu')
parser.add_argument('--decoder', type=str,default='ms',
                    help='decoder: ms sp')
parser.add_argument('--ds', type=str,
                    default='lfw',
                    help='The evaluation dataset')
parser.add_argument('--tau', type=float,
                    default=0.225,
                    help='nonce factor')
parser.add_argument('--z', type=int,
                    default=10,
                    help='Z factor')
parser.add_argument('--startiter', type=int,
                    default=0,
                    help='start iter number, default 0, 0-24')
parser.add_argument('--intervalnum', type=int,
                    default=4,
                    help='intervalnum')
parser.add_argument('--iters_max', type=int,
                    default=25,
                    help='iters_max')
parser.add_argument('--iters_interval', type=int,
                    default=1,
                    help='iters_interval')
parser.add_argument('--quanti_type', type=int,
                    default=1,
                    help='equal_probable:1; equal_space:2')


args = parser.parse_args()

# GPU settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


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


modelname = args.modelname
learning = args.learning
ds = args.ds


print('modelname:',modelname,'ds:',ds,'learning:',learning)
embeddings_o = np.loadtxt('embeddings/'+modelname+'.csv', delimiter=',') 
issame = np.loadtxt('embeddings/'+ds+'_issame.csv', delimiter=',') > 0

if ds == 'cfp_fp' or ds =='cfp_ff':
    embeddings_o[-12:,:] = embeddings_o[-24:-12,:] # https://github.com/ZhaoJ9014/face.evoLVe/issues/184

if args.quanti_type == 1:
    intervals = equal_probable(embeddings_o,intervalnum=args.intervalnum)
else:
    intervals = equal_space(embeddings_o,intervalnum=args.intervalnum)# check, the code may not be correct
    
print(intervals,intervals.shape)
if args.intervalnum == 2: # this is just sign it
    embeddings = (embeddings_o>0) * 1
else:
    embeddings = lssc_binary(embeddings_o,interval = intervals)

###
print("#####orignal acc eucliden######")
tpr, fpr, accuracy, best_thresholds = evaluate(embeddings_o, issame, 10)
d_p, genmean,impmean = d_prime(embeddings_o,issame,dist='eucliden', plot='plots/'+modelname+'_eucliden.svg')
print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)
print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = ['eucliden', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.std(genmean),np.mean(impmean),np.std(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')
    
print("#####orignal acc cosine######")
tpr, fpr, accuracy, best_thresholds = evaluate(embeddings_o, issame, 10)
d_p, genmean,impmean  = d_prime(embeddings_o,issame,dist='cosine', plot='plots/'+modelname+'_cosine.svg')
print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)
print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = ['cosine', accuracy.mean(),accuracy.std(),d_p,np.mean(genmean),np.std(genmean),np.mean(impmean),np.std(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')
    
print("#####orignal sign acc######")
tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_o>0, issame, 10)
d_p, genmean,impmean  = d_prime(embeddings_o>0,issame,dist='hamming', plot='plots/'+modelname+'_sign_hamming.svg')
print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)
print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = ['sign', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.std(genmean),np.mean(impmean),np.std(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')
    
print("#####binary acc######")
tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings, issame, 10)
d_p, genmean,impmean  = d_prime(embeddings,issame,dist='hamming', plot='plots/'+modelname+'_lssc_'+str(args.intervalnum)+str(args.quanti_type)+'hamming.svg')
print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p)
print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = ['lssc_binary', args.intervalnum, accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.std(genmean),np.mean(impmean),np.std(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')
    
print("#####look4noncerate######")
# look4noncerate(embeddings,issame,threshold = 0.15, genorimp=1,confidence = 0.95)
### embeddings_nonce,crossover_prob,nonce = look4noncerate(embeddings,issame,threshold = 0.15, genorimp=1,confidence = 0.95)# reply on ecc
embeddings_nonce,crossover_prob,nonce = look4noncerate(embeddings,issame,threshold = args.tau, genorimp=0,confidence = 0.95)# 95% imposters scores above 0.25
tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce, issame, 10)
d_p, genmean,impmean  = d_prime(embeddings_nonce,issame,dist='hamming', plot='plots/'+modelname+'_nonce_'+str(args.intervalnum)+str(args.quanti_type)+'hamming.svg')
print('crossover_prob',crossover_prob,',accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p, np.mean(genmean),np.mean(impmean))
print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = ['addnonce',crossover_prob, args.tau, accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.std(genmean),np.mean(impmean),np.std(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')

# print("#####xor user key######")
# embeddings_nonce_xorkey = adduserspecfkey(embeddings_nonce,issame)
# tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce_xorkey, issame, 10)
# d_p, genmean,impmean  = d_prime(embeddings_nonce_xorkey,issame,dist='hamming', plot='plots/'+modelname+'_xor_hamming.svg')
# print('accuracy:',accuracy.mean(),"±", accuracy.std(),'d_prime',d_p, np.mean(genmean),np.mean(impmean))
# print("far0:",tpr[np.where(fpr>0)[0][0]],"far001:",tpr[np.where(fpr>0.001)[0][0]])

# with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
#     val_list = ['addnonce n xor', accuracy.mean(),accuracy.std(),d_p, np.mean(genmean),np.mean(impmean),tpr[np.where(fpr>0)[0][0]],tpr[np.where(fpr>0.001)[0][0]]]
#     log = '\t'.join(str(value) for value in val_list)
#     f.writelines(log + '\n')
##########################################################
emb1 = embeddings_nonce[0::2]==0
emb2 = embeddings_nonce[1::2]==0


bio_noise = np.logical_xor(emb1,emb2)*1  # same -> 0 

gen = bio_noise[issame==1]
imp = bio_noise[issame==0]

bio_noise = -2*bio_noise+1
                
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    val_list = [args.decoder, args.modelname,args.ds, args.learning,args.tau,args.quanti_type,args.z,args.intervalnum]
    log = '\t'.join(str(value) for value in val_list)
    f.writelines(log + '\n')



def create_biometric_batch(wordRandom, noiseRandom, numOfWordSim, code_n, code_k, channel, Z, proto_ldpc, is_zeros_word, issamefalg = 1, fold = 0, blockindex = 0):# 500 * 6 blockindex: we split binary to several blocks, e.g., 1536 -> 512 * 3, each time we use 512 for 520 z=10 decoding, pading with 8 zeros
    X = np.zeros([1, code_n * Z], dtype=np.float32)
    Y = np.zeros([1, code_n * Z], dtype=np.int64)

    # build set for epoch
    # is_zeros_word = True
    if is_zeros_word:
        infoWord_i = 0 * wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z))
    else:
        infoWord_i = wordRandom.randint(0, 2, size=(numOfWordSim, code_k * Z)) # 520 bit for z=10
        
    # infoWord_i = Modulation.BPSK(infoWord_i)
    u_coded_bits = proto_ldpc.encode_LDPC(infoWord_i)
    s_mod = Modulation.BPSK(u_coded_bits)
    
    # noise_bsc = np.sign(np.random.random(size=s_mod.shape)-crossover_prob)
    
    # same_noise = bio_noise[issame==issamefalg,:code_n * Z]
    same_noise = bio_noise[issame==issamefalg, blockindex*512:(blockindex+1)*512] # its length is 512 * n, depend on the LSSC
    # pad 8 bits 
    paddings = np.ones((same_noise.shape[0],code_n * Z-same_noise.shape[1]))
    same_noise = np.hstack((same_noise, paddings))

    noise_bsc = same_noise[fold*500:(fold+1)*500,:]
    
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
# block_nums = 3
if args.z == 16:
    Z = args.z
    block_nums = 2
elif args.z == 10:
    Z = args.z
    block_nums = len(intervals) # match with LSSC 

proto_ldpc = Encode.Proto_LDPC(N, m, Z)
channel = Channel.BSC(N*Z)


##################################  Test  ####################################
sess.run(tf.global_variables_initializer())

iterss = [i for i in range(25)] + [i for i in range(29,100,10)]
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
totalfold = 6 # pay attention, we use 500 as a batch, here does not related to ten-fold 
if ds == 'cfp_fp' or ds =='cfp_ff':
    totalfold = 7
t = time.time()
hitss = np.zeros((len(iterss),totalfold))
hitss_imposter = np.zeros((len(iterss),totalfold))
for fold in range(totalfold):
    GT = np.zeros([batch_size, code_n * Z * block_nums], dtype=np.float32)
    Pred = np.zeros([len(iterss), batch_size, code_n * Z * block_nums], dtype=np.float32) # pay attention
    for blockindex in range(block_nums): # bio data is split into blocks 
        training_received_data, training_coded_bits = create_biometric_batch(wordRandom, noiseRandom, batch_size,
                                                                       code_n, code_k, channel,Z,
                                                                       proto_ldpc,
                                                                       train_on_zero_word, issamefalg = 1, fold = fold, blockindex = blockindex)
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


for fold in range(totalfold):
    GT = np.zeros([batch_size, code_n * Z * block_nums], dtype=np.float32)
    Pred = np.zeros([len(iterss), batch_size, code_n * Z * block_nums], dtype=np.float32) # pay attention
    for blockindex in range(block_nums): # bio data is split into blocks 
        training_received_data, training_coded_bits = create_biometric_batch(wordRandom, noiseRandom, batch_size,
                                                                       code_n, code_k, channel,Z,
                                                                       proto_ldpc,
                                                                       train_on_zero_word, issamefalg = 0, fold = fold, blockindex = blockindex)
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
        hitss_imposter[ttt,fold] = np.sum(hit>0.9999999999999)

        
print(hitss.shape,np.sum(hitss,axis=1))
print(hitss_imposter.shape,np.sum(hitss_imposter,axis=1))

gen_hits = np.sum(hitss,axis=1)
imp_hits = np.sum(hitss_imposter,axis=1)

elapsed = time.time() - t

with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    log = '\t'.join(str(value) for value in gen_hits)
    f.writelines(log + '\n')
with open('logs/log_ms_bsc_full_nonce_'+modelname+'_'+str(learning)+'.txt', 'a') as f:
    log = '\t'.join(str(value) for value in imp_hits)
    f.writelines(log + '\n')
########################################################################################
