from scipy.stats import norm
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math
import tqdm
from scipy.spatial.distance import pdist
from utils.verification import evaluate,evaluate_binary

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

def look4noncerate_full(embeddings,issame, genorimp=1, fileremark = 'embeddings_magface_ir100_calfw' ):##genorimp = 1 or 0
    if genorimp == 1:
        for crossover_prob in tqdm.tqdm(np.arange(0.0, 0.99, 0.005)):
            embeddings_nonce, nonce = addNonce(embeddings,crossover_prob=crossover_prob)
            d_p, gens, imps = d_prime(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            print(np.sum(imps >= threshold)/len(gens))
            # np.save('tmp/'+fileremark+'_nonce'+str(crossover_prob)+'.npy',embeddings_nonce)
            # np.save('tmp/'+fileremark+'_nonce_imps'+str(crossover_prob)+'.npy',imps)           
    else:
         for crossover_prob in tqdm.tqdm(np.arange(0.99,0.0,-0.005)):
            embeddings_nonce, nonce = addNonce(embeddings,crossover_prob=crossover_prob)
            d_p, gens, imps = d_prime(embeddings_nonce,issame,dist='hamming')# attention, gens are distance , only use hamming pls
            print(np.sum(imps >= 0.15)/len(gens))
#             np.save('tmp/'+fileremark+'_nonce'+str(crossover_prob)+'.npy',embeddings_nonce)
#             np.save('tmp/'+fileremark+'_nonce_imps'+str(crossover_prob)+'.npy',imps)
            

ds  = ['calfw','agedb_30','cfp_ff']
model = ['magface_ir100','vits_p12s8','arcface_r100','adaface_ir101_webface12m']

for d in tqdm.tqdm(ds):
    for mo in tqdm.tqdm(model):
        print('modelname:',mo,'ds:',d)
        embeddings_o = np.loadtxt('embeddings/embeddings_'+mo+'_'+d+'.csv', delimiter=',') 
        issame = np.loadtxt('embeddings/'+d+'_issame.csv', delimiter=',') > 0
        if ds == 'cfp_fp' or ds =='cfp_ff':
            embeddings_o[-12:,:] = embeddings_o[-24:-12,:] # https://github.com/ZhaoJ9014/face.evoLVe/issues/184
        look4noncerate_full(embeddings_o,issame, genorimp=0, fileremark = 'embeddings_'+mo+'_'+d )
