# 2026.3 월 말 부터 사용 중인 FER / BER 그래프 

import matplotlib.pyplot as plt
import numpy as np


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


plt.figure(figsize=(10, 7))

# semilogy
#plt.semilogy(SNR, BER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS - spatial weight sharing,  init alpha =0.7 init beta = 0.05")
plt.semilogy(SNR, FER_1, marker='o', markersize=6, linewidth=1.5,label=" NMS - edge weight sharing,  init alpha =0.7 init beta = 0.2")
plt.semilogy(SNR, BER_1,linestyle='--', marker='o', markersize=6, linewidth=1.5,label=" NMS - edge weight sharing,  init alpha =0.7 init beta = 0.2")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Frame Error Rate (FER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: test SNR\n SNR - BER ', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()