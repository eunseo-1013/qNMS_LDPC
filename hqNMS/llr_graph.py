import matplotlib.pyplot as plt
import numpy as np

# 데이터 설정
snr_axis = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
llr_min  = np.array([-11.15, -12.64, -12.56, -13.80, -14.06, -16.29, -184.88, -579.54, -686.47])
llr_max  = np.array([11.32, 12.50, 12.51, 15.02, 14.80, 16.26, 178.20, 564.08, 686.51])

# 스타일 초기화 (기본 화이트 테마)
plt.rcdefaults()
fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')

x = np.arange(len(snr_axis))
bar_width = 0.5

# Floating Bar: Min에서 Max까지 이어지는 기둥
# 색상은 신뢰감을 주는 밝은 블루 계열
bars = ax.bar(x, (llr_max - llr_min), bottom=llr_min, width=bar_width, 
              color='#60A5FA', edgecolor='#2563EB', linewidth=1.5, 
              alpha=0.7, label='LLR Range')

# 0점 기준선 (데이터의 중심 확인용)
ax.axhline(0, color='black', linewidth=1.2, linestyle='-')

# 가독성을 위한 보조 지표 (Max, Min 점 찍기)
ax.scatter(x, llr_max, color='#1E40AF', s=50, zorder=3, label='Max Value')
ax.scatter(x, llr_min, color='#B91C1C', s=50, zorder=3, label='Min Value')

# 축 설정
ax.set_xticks(x)
ax.set_xticklabels([f'{s}dB' for s in snr_axis], fontsize=11, fontweight='bold')
ax.set_ylabel('LLR Value', fontsize=12, fontweight='bold')
ax.set_title(' LLR Dynamic Range Comparison (SNR)', fontsize=16, pad=20, fontweight='bold')

# 그리드 및 배경 깔끔하게 정리
ax.grid(axis='y', linestyle='--', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 수치 텍스트 (SNR 4.0 이상 큰 값들 위주로 강조)
for i in range(len(snr_axis)):
    if abs(llr_max[i]) > 100: # 큰 값만 텍스트 표시해서 깔끔하게
        ax.text(i, llr_max[i] + 15, f'{llr_max[i]:.0f}', ha='center', color='#1E40AF', fontweight='bold')
        ax.text(i, llr_min[i] - 40, f'{llr_min[i]:.0f}', ha='center', color='#B91C1C', fontweight='bold')

ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()