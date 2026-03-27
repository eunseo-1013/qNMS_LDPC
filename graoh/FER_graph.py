# 2026.3 월 말 부터 사용 중인 FER / BER 그래프 

import matplotlib.pyplot as plt
import numpy as np
from qNMS.NMS.graph import BER_4


# snr 별 train - test 한 것들
# 환경
# model 1 bit
frame = 100000  # 10^5
batch = 20
test_batch=50
epoch = 1
test_frame= 10000

iteration_num=20

train_snr=2.0 
learning_rate=0.001

#fixed
eta=0.5


SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]


BER_1=[0.1305902777777778, 0.1164412037037037, 0.10174120370370371, 0.08704907407407407, 0.07158032407407408, 0.05139375, 0.024727314814814813, 0.005506712962962963, 0.0004787037037037037]
BER_2=[0.14021712962962962, 0.1274287037037037, 0.11492939814814815, 0.10339120370370371, 0.09171296296296297, 0.08092407407407408, 0.07062083333333333, 0.0604875, 0.050522222222222225]
FER_1=[1.0, 1.0, 1.0, 1.0, 0.9982, 0.9313, 0.5894, 0.1673, 0.0187]
FER_2=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]



BER_2=[0.037529539351851854,0.029558233796296296, 0.015618256944444445,0.006685800925925926,0.0048852453703703705,0.0006499444444444444]
FER_2=[1.0,0.999992, 0.98085,0.843103,0.674187,0.333813,0.204086 ]


BER_3=[0.03486280555555556, 0.020426039351851853, 0.00946644212962963, 0.0014520787037037037, 0.0003392175925925926, 0.00014804398148148147, 3.9655092592592596e-05]
BER_4=[0.999533, 0.988919, 0.920302, 0.378916, 0.122786, 0.057432, 0.016241]

plt.figure(figsize=(10, 7))

# semilogy
#plt.semilogy(SNR, BER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS - spatial weight sharing,  init alpha =0.7 init beta = 0.05")
plt.semilogy(SNR, BER_1, color='light grey',marker='o', markersize=6, linewidth=1.5,label="NMS")
plt.semilogy(SNR, FER_1,color='light grey',linestyle='--', marker='o', markersize=6, linewidth=1.5,label="NMS")


plt.semilogy(SNR, BER_2,color='red', marker='o', markersize=6, linewidth=1.5,label=" all 2bit qNMS")
plt.semilogy(SNR, FER_2,color='red',linestyle='--', marker='o', markersize=6, linewidth=1.5,label="all 2bit qNMS")


plt.semilogy(SNR, BER_3,color='orange', marker='o', markersize=6, linewidth=1.5,label=" all 3bit qNMS")
plt.semilogy(SNR, FER_3,color='orange',linestyle='--', marker='o', markersize=6, linewidth=1.5,label="all 3bit qNMS")


plt.semilogy(SNR, BER_4,color='blue', marker='o', markersize=6, linewidth=1.5,label=" all 6bit qNMS")
plt.semilogy(SNR, FER_4,color='blue',linestyle='--', marker='o', markersize=6, linewidth=1.5,label="all 6bit qNMS")


plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Frame Error Rate (FER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: test SNR\n SNR - BER ', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()