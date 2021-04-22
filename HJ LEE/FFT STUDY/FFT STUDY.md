# Discrete Signal Fourier Transform

NumPy에서는 numpy.fft에서 Matlab과 거의 동일한 형식으로 DFT를 지원,
주로 사용하는 함수는 fft(x), ifft(x), fftfreq(n), fftshift(x) 등
또한 matplotlib의 stem(x)를 stem plot을 그리는 데 많이 사용함
FFT의 정의는 Matlab과 동일하다. 다만 인덱스가 0부터라는 점에 주의해야 함


forward Fourier Transform X = fft(x)*dt


## sample 1 : DFT 주파수 분석 예시
https://wikidocs.net/14635

60 Hz 와 120 Hz의 사인 곡선이 중첩된 신호
x(t)=0.7sin(120πt)+sin(240πt)
에 N(μ=0,σ=2)를 따르는 노이즈가 포함된 신호에 대해 주파수 분석을 한 예

샘플링 주파수는 10kHz이고 200000개의 데이터로 수정해서 진행

### code

import numpy as np
import matplotlib.pyplot as plt

fs = 1000*10     # sampling frequency 10kHz
dt = 1/fs     # sampling period
N  = 20000     # length of signal

t  = np.arange(0,N)*dt   # time = [0, dt, ..., (N-1)*dt]

s = 0.7*np.sin(2*np.pi*60*t) + np.sin(2*np.pi*120*t)

x = s+2*np.random.randn(N)   # random number Normal distn, N(0,2)... N(0,2*2)

plt.subplot(2,2,1)
plt.plot(t[0:501],s[0:501],label='s')
plt.legend()
plt.xlabel('time'); plt.ylabel('x(t)'); plt.grid()


plt.subplot(2,2,2)
plt.plot(t[0:501],x[0:501],label='noise')
plt.legend()
plt.xlabel('time'); plt.ylabel('x(t)'); plt.grid()

df = fs/N   # df = 1/N = fs/N
f = np.arange(0,N)*df     #   frq = [0, df, ..., (N-1)*df]

xf = np.fft.fft(x)*dt

sf = np.fft.fft(s)*dt

plt.subplot(2,2,3)

plt.plot(f[0:int(N/50+1)],np.abs(sf[0:int(N/50+1)]))
plt.xlabel('frequency(Hz)'); plt.ylabel('abs(xf)'); plt.title('s signal fft'); plt.grid()
plt.tight_layout()


plt.subplot(2,2,4)
plt.plot(f[0:int(N/50+1)],np.abs(xf[0:int(N/50+1)]))
plt.xlabel('frequency(Hz)'); plt.ylabel('abs(xf)'); plt.title('s + noise signal fft');plt.grid()
plt.tight_layout(rect=(0,0,2,2))




## sample 2 : python EEG FFT sample 
https://dsp.stackexchange.com/questions/45345/how-to-correctly-compute-the-eeg-frequency-bands-with-python

### code

import numpy as np

fs = 512                                # Sampling rate (512 Hz)
data = np.random.uniform(0, 100, 1024)  # 2 sec of data b/w 0.0-100.0

####Get real amplitudes of FFT (only in postive frequencies)
fft_vals = np.absolute(np.fft.rfft(data))

#### Get frequencies for amplitudes in Hz
fft_freq = np.fft.rfftfreq(len(data), 1.0/fs)

#### Define EEG bands
eeg_bands = {'Delta': (0, 4),
             'Theta': (4, 8),
             'Alpha': (8, 12),
             'Beta': (12, 30),
             'Gamma': (30, 45)}

#### Take the mean of the fft amplitude for each EEG band
eeg_band_fft = dict()
for band in eeg_bands:  
    freq_ix = np.where((fft_freq >= eeg_bands[band][0]) & 
                       (fft_freq <= eeg_bands[band][1]))[0]
    eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

#### Plot the data (using pandas here cause it's easy)
import pandas as pd
df = pd.DataFrame(columns=['band', 'val'])
df['band'] = eeg_bands.keys()
df['val'] = [eeg_band_fft[band] for band in eeg_bands]
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
