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




# Computing the power spectral density
https://raphaelvallat.com/bandpower.html
raw EEG data와 예제 sample code 있음

In order to compute the average bandpower in the delta band, we first need to compute an estimate of the power spectral density. 
The most widely-used method to do that is the "Welch's periodogram", 
which consists in averaging consecutive Fourier transform of small windows of the signal, with or without overlapping.

The Welch's method improves the accuracy of the classic periodogram. 
The reason is simple: EEG data are always time-varying, meaning that if you look at a 30 seconds of EEG data, 
it is very (very) unlikely that the signal will looks like a perfect sum of pure sines. 

Rather, the spectral content of the EEG changes over time, constantly modified by the neuronal activity at play under the scalp. 
Problem is, to return a true spectral estimate, a classic periodogram requires the spectral content of the signal to be stationnary (i.e. time-unvarying) over the time period considered. 

Because it is never the case, the periodogram is generally biased and contains way too much variance (see the end of this tutorial). 
By averaging the periodograms obtained over short segments of the windows, the Welch's method allows to drastically reduce this variance. 
This comes at the cost, however, of a lower frequency resolution. Indeed, the frequency resolution is defined by:

  Fres=Fs/N=Fs/Fst=1/t

where Fs is the sampling frequency of the signal, N the total number of samples and t the duration, in seconds, of the signal. In other words, if we were to use the full length of our data (30 seconds), our final frequency resolution would be 
1/30 = 0.033 Hz, which is 30 frequency bins per Hertz. By using a 4-second sliding window, we reduce this frequency resolution to 4 frequency bins per Hertz, i.e. each step represents 0.25 Hz.

How do we define the optimal window duration then? A commonly used approach is to take a window sufficiently long to encompasses at least two full cycles of the lowest frequency of interest. In our case, our lowest frequency of interest is 0.5 Hz so we will choose a window of 
2/0.5=4 seconds.