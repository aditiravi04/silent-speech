import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

# normalize data
def normalized_fft(signal, sampleRate):
    N = len(signal)          # sample pts
    df = sampleRate / N
    f = np.arange(N) * df    
    yf = fft(signal)
    magnitude = np.fft.fftshift(np.fft.fft(signal))
    magnitude = np.abs(magnitude) #take abs
    #magnitude = 2.0/N * np.abs(yf[:N//2])
    f[f >= sampleRate / 2] -= sampleRate
    frequency = np.fft.fftshift(f) # freq
    return frequency, magnitude / np.mean(magnitude) #ret vals, take mean of magnitude to normalize

def convert_to_numeric(data):
    # str->numeric, drop invalid
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            
        data = data.dropna(axis = 0, inplace = True)
        
        return data
    
def plot(csv_file, sampleRate):

    data = pd.read_csv(csv_file)
    
    convert_to_numeric(data) #str->numeric
    
    # accel
    plt.figure(figsize=(12, 6))
    #for accel cols
    for col in ['acce_x', 'acce_y', 'acce_z']:
        #normalize_fft
        frequency, mag_normalized = normalized_fft(data[col].values, sampleRate)
        #plot
        plt.plot(frequency, mag_normalized, label=col)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (Normalized)')
    plt.title('FFT of Accelerometer Data')
    plt.legend()
    plt.grid()
    plt.xlim(0,100)
    plt.ylim(-50, 200)

    # gyro
    plt.figure(figsize=(12, 6))
    #gyro cols
    for col in ['gyro_x', 'gyro_y', 'gyro_z']:
        #normalize_fft
        frequency, mag_normalized = normalized_fft(data[col].values, sampleRate)
        #plot
        plt.plot(frequency, mag_normalized, label=col)
    
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (Normalized)')
    plt.title('FFT of Gyroscope Data')
    plt.legend()
    plt.grid()
    plt.xlim(0,200)
    #plt.ylim(-100, 400)
    
    plt.show()


if __name__ == '__main__':
    plot('up/up_04.csv', 1000) #replace w/ whichever file