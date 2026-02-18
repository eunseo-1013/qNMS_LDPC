import matplotlib.pyplot as plt
import numpy as np
SNR = [1.0, 1.5, 2.0, 2.5, 3.0,3.5, 4.0,4.5, 5.0]


BER_1 =[0.13630520833333334, 0.12211267361111111, 0.1074515625, 0.09266770833333333, 0.07566145833333333, 0.05368836805555555, 0.025624652777777777, 0.005802951388888889, 0.0004840277777777778]
BER_2 = [0.13066128472222222, 0.11636944444444444, 0.10228125, 0.08797118055555556, 0.07314583333333333, 0.055771006944444444, 0.032215451388888885, 0.009335416666666667, 0.0010921875]
BER_3 =[0.13067239583333334, 0.11637725694444444, 0.10228454861111111, 0.0879828125, 0.07314045138888889, 0.05557170138888889, 0.03185173611111111, 0.009530381944444444, 0.0013815972222222222]
BER_4=[0.1327706597222222, 0.11830277777777778, 0.103790625, 0.08914201388888889, 0.07328472222222222, 0.05656111111111111, 0.039091145833333334, 0.023504513888888887, 0.012948958333333333]
BER_q1=[0.13108159722222223, 0.11748958333333333, 0.10442013888888889, 0.09139409722222222, 0.07886631944444444, 0.06733854166666667, 0.05669791666666667, 0.046489583333333334, 0.03790798611111111]
BER_5=[0.13256597222222222, 0.11809652777777778, 0.10352152777777777, 0.08902083333333333, 0.07366753472222222, 0.05799861111111111, 0.04210572916666667, 0.02771909722222222, 0.016755208333333334]
BER_6=[0.13057673611111112, 0.11659166666666666, 0.102284375, 0.08790972222222222, 0.0732390625, 0.05720243055555556, 0.04025190972222222, 0.02317326388888889, 0.010581944444444444]

frame = 5000
batch = 50
epoch = 10
test_frame= 10000

iteration_num=20

train_snr=2.0 



# 로그 그래프

plt.figure(figsize=(10, 7))

# semilogy를 사용하면 Y축이 자동으로 로그 스케일이 됩니다.
plt.semilogy(SNR, BER_1, marker='o', markersize=6, linewidth=1.5,label=" MS , fixed alpha =0.7 fixed beta = 0.2")
plt.semilogy(SNR, BER_2, marker='o', markersize=6, linewidth=1.5,label=" NMS , init alpha =0.7 init beta = 0.2")
plt.semilogy(SNR, BER_3, marker='o', markersize=6, linewidth=1.5,label=" NMS , init alpha =0.7 init beta = 0.05")
plt.semilogy(SNR, BER_4, marker='o', markersize=6, linewidth=1.5,label=" 2bit qNMS -2~2")
plt.semilogy(SNR, BER_5, marker='o', markersize=6, linewidth=1.5,label=" 4bit qNMS -2~2")
plt.semilogy(SNR, BER_6, marker='o', markersize=6, linewidth=1.5,label=" 4bit qNMS -4~4")
plt.xlabel("SNR (dB)", fontsize=12)
plt.ylabel("Bit Error Rate (BER)", fontsize=12)
plt.title(f'Iteration: {iteration_num}, Train SNR: {train_snr}dB\n SNR - BER (Log Scale)', fontsize=14)

plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend(loc='best', fontsize=10)
plt.show()


# 일반그래프
'''




step=30
plt.figure()
plt.plot(x[::step], q_0[::step], label=f'shaping_factor=0.5')
plt.plot(x[::step], q_1[::step], label=f'shaping_factor=0.05')
plt.plot(x[::step], q_2[::step], label='shaping_factor=0.005')
plt.xlabel("x")
plt.ylabel("Qn(x)")
plt.legend()
plt.title(f'qk = [-1,-1/3,1/3,1] b=2')
plt.grid(True)
plt.show()

'''