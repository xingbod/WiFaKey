from scipy.stats import norm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import tqdm
from scipy.spatial.distance import pdist
from utils.verification import evaluate,evaluate_binary

def adduserspecfkey(embeddings_nonce,issame, worstcase=False):
    embeddings_nonce_xorkey = np.zeros(embeddings_nonce.shape)
    totl = int(embeddings_nonce.shape[0]/2)
    dim = embeddings_nonce.shape[1]
    for i in range(totl):
        if not issame[i]:
            nonce1 = np.random.random(size=dim)-0.5>0
            if worstcase:
                nonce2 = nonce1
            else:
                nonce2 = np.random.random(size=dim)-0.5>0
                
            embeddings_nonce_xorkey[2*i,:] = np.logical_xor(embeddings_nonce[2*i,:],nonce1)
            embeddings_nonce_xorkey[2*i+1,:] = np.logical_xor(embeddings_nonce[2*i+1,:],nonce2)
        else:
            nonce1 = np.random.random(size=dim)-0.5>0
            embeddings_nonce_xorkey[2*i,:] = np.logical_xor(embeddings_nonce[2*i,:],nonce1)
            embeddings_nonce_xorkey[2*i+1,:] = np.logical_xor(embeddings_nonce[2*i+1,:],nonce1)
    return embeddings_nonce_xorkey

def equal_probable(embeddings_o, intervalnum=4):
    meanv,stdv = np.mean(np.sum(embeddings_o,axis=1)/512), np.mean(np.std(embeddings_o,axis=1))
    print('mean: ',meanv,', std:', stdv)
    minv = np.min(embeddings_o)
    maxv = np.max(embeddings_o)
    startv = minv - 0.1
    interval = 1 / intervalnum
    intervals = []
    for i in range(intervalnum-1):
        for endv in np.arange(startv, maxv + 0.1, 0.0001):
            # endv = 0.03
            pro=norm(meanv, stdv).cdf(endv) - norm(meanv, stdv).cdf(startv)
            if pro>=interval:
                intervals.append(endv)
                startv = endv
                # print(intervals)
                break

        # print(pro)
    # print('solved:',endv,pro)
    assert len(intervals) == intervalnum-1
    return np.array(intervals)

def equal_space(embeddings_o, intervalnum=4):
    minv = np.min(embeddings_o)
    maxv = np.max(embeddings_o)
    step = (maxv-minv)/intervalnum
    intervals = []
    for i in range(1,intervalnum):
        intervals.append(minv+step*i)
        
    return np.array(intervals)


def onehot_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    # interval = np.array([-0.1,-0.05,-0.025,0,0.025, 0.05,0.1])
    # interval = np.array([-0.03, 0 , 0.03])
    lkut = np.zeros((len(interval)+1,len(interval)+1)) 
    for i in range(len(interval)+1):
        lkut[i,len(interval)-i] = 1
    print(lkut)
    block = len(interval) + 1
    def LSSC(data):
        new_data = np.zeros(512*block)
        for i in range(len(data)):
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])        
            new_data[i*block:(i+1)*block] = lkut[index]
        return new_data


    embeddings = np.zeros((len(embeddings_o),512*block))
    for i in tqdm.tqdm(range(len(embeddings_o))):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    
    return embeddings

def lssc_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    lkut = np.zeros((len(interval)+1,len(interval))) 
    for i in range(1,len(interval)+1):
        lkut[i,len(interval)-i:] = 1
    print(lkut)
    def LSSC(data):
        new_data = np.zeros(512*len(interval))
        for i in range(len(data)):
            # print(interval>data[i],data[i])
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])
            new_data[i*len(interval):(i+1)*len(interval)] = lkut[index]
        return new_data
    block = len(interval) 

    embeddings = np.zeros((len(embeddings_o),512*block))
    for i in tqdm.tqdm(range(len(embeddings_o))):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    return embeddings

def brgc_binary(embeddings_o,interval = np.array([-0.03, 0 , 0.03])):
    block = 2
    lkut = np.zeros((len(interval)+1,block)) 
    for i in range(1,len(interval)+1):
        lkut[i,:] = np.array( [int(s) for s in "{0:02b}".format(i)])

    print(lkut)

    def LSSC(data):
        new_data = np.zeros(512*block)
        for i in range(len(data)):
            index = -1
            whereindex = np.where(interval>data[i])
            if len(whereindex[0]) != 0:
                index = whereindex[0][0]
            # print(index,embeddings_o[0,i], np.where(interval>embeddings_o[0,i]),interval>embeddings_o[0,i])        
            new_data[i*block:(i+1)*block] = lkut[index]
        return new_data

    embeddings = np.zeros((12000,512*block))
    for i in tqdm.tqdm(range(12000)):
        embeddings[i,:] =  LSSC(embeddings_o[i,:])
    return embeddings


def d_prime_savefig(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
        Dprime= np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen*100, bins=100, label="Mated: {:.2%}".format(np.mean(gen)))
            plt.hist(imp*100, bins=100, label="Nonmated: {:.2%}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Hamming distance in percentage')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Cosine distance')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.show()
            fig.savefig(plot)

    
    return Dprime, gen, imp


def d_prime(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
        Dprime= np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen*100, bins=100, label="Mated: {:.2%}".format(np.mean(gen)))
            plt.hist(imp*100, bins=100, label="Nonmated: {:.2%}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Hamming distance in percentage')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Cosine distance')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
        Dprime = np.abs( np.mean(gen) - np.mean(imp)) / math.sqrt(0.5*(np.std(gen)**2+np.std(imp)**2))
        if plot:
            fig, ax = plt.subplots()
            plt.hist(gen, bins=100, label="Mated: {:.2f}".format(np.mean(gen)))
            plt.hist(imp, bins=100, label="Nonmated: {:.2f}".format(np.mean(imp)))
            plt.legend(loc="upper left")
            plt.xlabel('Euclidean distance')
            plt.ylabel('Frequency')
            plt.show(block=False)
            fig.savefig(plot)

    
    return Dprime, gen, imp

def computeGenImp(embeddings,issame,dist='hamming', plot=''):
    emb1 = embeddings[0::2]
    emb2 = embeddings[1::2]
    def get_cos_similar(v1: list, v2: list):
        num = float(np.dot(v1, v2))  # 向量点乘
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
        return 0.5 + 0.5 * (num / denom) if denom != 0 else 0
    if dist=='hamming':
        dist = np.logical_xor(emb1,emb2) * 1
        bio_noise = np.logical_xor(emb1,emb2) * 1
        gen = bio_noise[issame==1]
        imp = bio_noise[issame==0]
        gen = np.sum(gen,axis = 1)/embeddings.shape[1]
        imp = np.sum(imp,axis = 1)/embeddings.shape[1]
       
    elif dist=='cosine':
        dist = []
        for i in range(len(emb1)):
            dist.append(1-get_cos_similar(emb1[i,:], emb2[i,:]))
        dist = np.array(dist)
        gen = dist[issame==1]
        imp = dist[issame==0]

    else:
        diff = np.subtract(emb1, emb2)
        dist = np.sum(np.square(diff), 1)
        gen = dist[issame==1]
        imp = dist[issame==0]
  
    
    return gen, imp
def addNonce(embeddings,crossover_prob=0.30,seed = 2):
    # crossover_prob=0.30
    # np.random.seed(seed)# for duplicates 
    nonce = np.random.random(size=embeddings.shape[1])-crossover_prob>0
    embeddings_nonce = np.zeros(embeddings.shape)
    for i in range(len(embeddings)):
        embeddings_nonce[i,:] = np.logical_and(embeddings[i,:],nonce)
    return embeddings_nonce, nonce


'''
threshold = 0.15, genorimp=1

Strong face accept resistance 
threshold = 0.15, genorimp=2

'''
def look4noncerate(embeddings,issame,threshold = 0.15, genorimp=1,confidence = 0.95, start=0.99 ):##genorimp = 1 or 0
    if genorimp == 1:
        for crossover_prob in tqdm.tqdm(np.arange(0.0, 0.99, 0.005)):
            embeddings_nonce, nonce = addNonce(embeddings,crossover_prob=crossover_prob)
            # d_p, gens, imps = d_prime(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            gens, imps = computeGenImp(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            # tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce, issame, 10)
            if np.sum(gens <= threshold)/len(gens)>=confidence:
                break  
    else:
         for crossover_prob in tqdm.tqdm(np.arange(start,0.0,-0.005)):
            embeddings_nonce, nonce = addNonce(embeddings,crossover_prob=crossover_prob)
            # d_p, gens, imps = d_prime(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            gens, imps = computeGenImp(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            if np.sum(imps >= threshold)/len(gens)>=confidence:
                break 
    tpr, fpr, accuracy, best_thresholds = evaluate_binary(embeddings_nonce, issame, 10)
    # print('crossover_prob:',crossover_prob,',accuracy:',accuracy.mean(),"±", accuracy.std(),',d_prime',d_p)
    print('crossover_prob:',crossover_prob,',accuracy:',accuracy.mean(),"±", accuracy.std())
    return embeddings_nonce,crossover_prob, nonce

