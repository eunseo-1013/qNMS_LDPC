
import pandas as pd
import numpy as np
import torch

# H 파일 열기 ( 이미 shifting o )
#wman_N0576_R34_z24.txt
#wman_N1152_R12_z48.txt
#wman_N1152_R12_z72.txt


filename="wman_N0576_R34_z24.txt"
N=int(filename[6:10])
K=N*int(filename[12:13])/int(filename[13:14])
K=int(K)

print("N:", N ,", K :" , K)

def make_k_bit(K): # 돌려볼 비트 만들기 k 길이 짜리
   shape=(1,K)
   orignal_bit=torch.randint(low=0,high=2,size=shape,dtype=int)
   return (orignal_bit)


orignal_bit=make_k_bit(K)

def H_to_tensor(filename):
    df=pd.read_csv(filename,header=None,sep='\s+')
    np_array=df.values.astype(np.int64)
    tr=torch.tensor(np_array)
    print("H shape :", tr.shape)
    return tr

H=H_to_tensor(filename)

def RREF(H): ## 이진연산을 써야함!!!!!!! 아오 바보야 ㅠㅠ H[ A | I ]
    pivot_row=0
    for col in range(H.shape[1]-H.shape[0],H.shape[1]):
    
        #상대적 위치를 찾는다. <- tuple ( tensor , )
        find_1=torch.where((H[pivot_row: , col] ==1))[0] # = tensor ( 1 이 있는 인덱스)
        if(len(find_1)==0): # 1인 인덱스가 x == 이 열이 전부 0 일경우... 패쓰
            continue
        max_idx=find_1[0].item()+pivot_row # 텐서 일반 숫자로 변경 필요
       
        H[[pivot_row,max_idx]]=H[[max_idx,pivot_row]] #swap
        
        # H[pivot_row]= H[pivot_row]/H[pivot_row,col]  <- xor 연산이기에 0,1 만 존재 == 1로만들필요 x
        for i in range(H.shape[0]):
            if(i!=pivot_row and H[i,col] == 1):
                 H[i]=H[i]^H[pivot_row]

        pivot_row+=1
    return H




def make_G_using_H(RREF_H,K):

    I=torch.eye(K,dtype=RREF_H.dtype)
    A=RREF_H[:,:K]
    print(A.shape)
    G=torch.cat([I,A.t()],dim=1)
    print("G shape:",G.shape)
    return G

G=make_G_using_H(RREF(H),K)

print("original _code ",orignal_bit.shape)
ldpc_code=orignal_bit@G
ldpc_code=ldpc_code%2
print(ldpc_code.shape)
print("인코딩 잘 됐나~?:", torch.equal(orignal_bit, ldpc_code[:, :K].long()))
#print(ldpc_code)
Z=(H@ldpc_code.T)
Z=Z%2
print(Z.shape)





