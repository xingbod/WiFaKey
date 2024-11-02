import numpy as np

# bch
from sympy import Poly
from sympy.abc import x, alpha
from bch.bchcodegenerator import BchCodeGenerator
from bch.bchcoder import BchCoder
from bch.padding import padding_encode,padding_decode

class LDPC:
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        self.G_matrix, self.H_matrix = self.init_LDPC_G_H(file_G, file_H)

    def init_LDPC_G_H(self, file_G, file_H):
        G_matrix_row_col = np.loadtxt(file_G, dtype=np.int32)
        H_matrix_row_col = np.loadtxt(file_H, dtype=np.int32)
        G_matrix = np.zeros([self.K, self.N], dtype=np.int32)
        H_matrix = np.zeros([self.N-self.K, self.N], dtype=np.int32)
        G_matrix[G_matrix_row_col[:, 0], G_matrix_row_col[:, 1]] = 1
        H_matrix[H_matrix_row_col[:, 0], H_matrix_row_col[:, 1]] = 1
        return G_matrix, H_matrix
    
    def encode_LDPC(self, x_bits):
        u_coded_bits = np.mod(np.matmul(x_bits, self.G_matrix), 2)
#        check = np.mod(np.matmul(u_coded_bits, np.transpose(self.H_matrix)),2)
        u_coded_bits = (-1)**u_coded_bits
        return u_coded_bits
        
    def dec_src_bits(self, bp_output):
        return bp_output[:,0:self.K]                 

'''
循环码是一类重要的线性分组码, 码字具有循环移位性质, 即若 $C=\left[c_{n-1} c_{n-2} \cdots c_{1} c_{0}\right]$ 是 某循环码的码字, 那么码字 $C$ 的所有循环移位都是码字。现实中许多重要的分组码都具有 循环性。循环码有以下两大特点:
第一, 循环码具有严谨的代数结构, 可以找到各种实用的译码方法;
第二, 由于其循环特性, 编码运算和伴随式计算, 可用反馈移位寄存器来实现, 电路 实现简单。
'''
class RS:
    def __init__(self, N, K, file_G, file_H):
        self.N = N
        self.K = K
        

class BCH:

    def __init__(self, n, b, d):
        self.n = n
        self.b = b
        self.d = d
        bch_gen = BchCodeGenerator(n, b, d)
        r, g = bch_gen.gen()
        # np.savez_compressed(code_file, n=n, b=b, d=d, r=r.all_coeffs()[::-1], g=g.all_coeffs()[::-1])
        # log.info("BCH code saved to {} file".format(code_file))
        self.r = r
        self.g = g

    def encode(self, input_arr, block=False):
        code = np.load(code_file, allow_pickle=True)
        bch = BchCoder(self.n, self.b, self.d, Poly(self.r[::-1], alpha),
                       Poly(self.g[::-1], x))
        if not block:
            if len(input_arr) > bch.k:
                raise Exception("Input is too large for current BCH code (max: {})".format(bch.k))
            return bch.encode(Poly(input_arr[::-1], x))[::-1]

        input_arr = padding_encode(input_arr, bch.k)
        input_arr = input_arr.reshape((-1, bch.k))
        output = np.array([])
        block_count = input_arr.shape[0]
        for i, b in enumerate(input_arr, start=1):
            log.info("Processing block {} out of {}".format(i, block_count))
            next_output = np.array(bch.encode(Poly(b[::-1], x))[::-1])
            if len(next_output) < bch.n:
                next_output = np.pad(next_output, (0, bch.n - len(next_output)), 'constant')
            output = np.concatenate((output, next_output))
        return output


    def decode(self, input_arr, block=False):
        bch = BchCoder(self.n, self.b, self.d, Poly(self.r[::-1], alpha),
                       Poly(self.g[::-1], x))
        if not block:
            if len(input_arr) > bch.n:
                raise Exception("Input is too large for current BCH code (max: {})".format(bch.n))
            return bch.decode(Poly(input_arr[::-1], x))[::-1]

        input_arr = input_arr.reshape((-1, bch.n))
        output = np.array([])
        block_count = input_arr.shape[0]
        for i, b in enumerate(input_arr, start=1):
            log.info("Processing block {} out of {}".format(i, block_count))
            next_output = np.array(bch.decode(Poly(b[::-1], x))[::-1])
            if len(next_output) < bch.k:
                next_output = np.pad(next_output, (0, bch.k - len(next_output)), 'constant')
            output = np.concatenate((output, next_output))

        return padding_decode(output, bch.k)


class Proto_LDPC:
    def __init__(self, N, m, Z):
        self.code_n = N
        self.Z = Z
        self.code_k = N - m
        # get the base graph and generator matrix
        code_PCM0 = np.loadtxt("./LDPC_MetaData/BaseGraph/BaseGraph2_Set0.txt", int, delimiter='	')
        code_PCM1 = np.loadtxt("./LDPC_MetaData/BaseGraph/BaseGraph2_Set1.txt", int, delimiter='	')
        code_PCM2 = np.loadtxt("./LDPC_MetaData/BaseGraph/BaseGraph2_Set2.txt", int, delimiter='	')
        code_GM_16 = np.loadtxt("./LDPC_MetaData/BaseGraph_GM/LDPC_GM_BG2_16.txt", int, delimiter=',')
        code_GM_3 = np.loadtxt("./LDPC_MetaData/BaseGraph_GM/LDPC_GM_BG2_3.txt", int, delimiter=',')
        code_GM_10 = np.loadtxt("./LDPC_MetaData/BaseGraph_GM/LDPC_GM_BG2_10.txt", int, delimiter=',')
        code_GM_6 = np.loadtxt("./LDPC_MetaData/BaseGraph_GM/LDPC_GM_BG2_6.txt", int, delimiter=',')
        self.Ldpc_PCM = [code_PCM0, code_PCM1, code_PCM2, code_PCM1]# four LDPC codes with different code lengths
        self.Ldpc_GM = [code_GM_16, code_GM_3, code_GM_10, code_GM_6]
        if Z == 16:
            self.G_matrix = self.Ldpc_GM[0]
        elif Z==3:
            self.G_matrix = self.Ldpc_GM[1]
        elif Z==10:
            self.G_matrix = self.Ldpc_GM[2]
        elif Z==6:
            self.G_matrix = self.Ldpc_GM[3]


    def encode_LDPC(self, x_bits):
        # u_coded_bits = np.mod(np.matmul(x_bits, self.G_matrix), 2)
        u_coded_bits = np.dot(x_bits, self.G_matrix) % 2
#        check = np.mod(np.matmul(u_coded_bits, np.transpose(self.H_matrix)),2)
        # u_coded_bits = (-1)**u_coded_bits
        return u_coded_bits
        
    def dec_src_bits(self, bp_output):
        return bp_output[:,0:self.code_k]                 

