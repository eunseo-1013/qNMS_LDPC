import matplotlib.pyplot as plt
import numpy as np
SNR = [1.0, 2.0, 3.0, 4.0, 5.0]


BER_1 = [0.14791665971279144, 0.11267361044883728, 0.0963541641831398, 0.08697916567325592, 0.0703125]
BER_2 = [0.1597222238779068, 0.09375, 0.1059027761220932, 0.0625, 0.1180555522441864]
BER_3 =[0.1597222238779068, 0.1336805522441864, 0.1145833358168602, 0.1232638880610466, 0.1440972238779068]


plt.figure()
plt.plot(SNR, BER_1, marker='o', label='iteration=3')
plt.plot(SNR, BER_2, marker='s', label='iteration=5')
plt.plot(SNR, BER_3, marker='^', label='iteration=8')
plt.xlabel("SNR (db)")
plt.ylabel("BER")
plt.legend()
plt.title("SNR - BER")
plt.grid(True)
plt.show()