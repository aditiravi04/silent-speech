import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

    
def spectro_plot(data, sampleRate, name):
    plt.figure(figsize=(10,6))
    for r, c in enumerate(data.columns):
        plt.subplot(len(data.columns), 1, r+1)
        f, t, Sxx = signal.spectrogram(data[c], sampleRate, nperseg=64, noverlap=10)
        plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud')
        plt.ylabel(f'{c} Frequency [Hz]')
        plt.colorbar(label='Intensity [dB]')
    plt.xlabel('Time [s]')
    plt.suptitle(name)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()

if __name__ == '__main__':
    data = pd.read_csv('data/up_01.csv')
    acce_data = data[['acce_x', 'acce_y', 'acce_z']]
    gyro_data = data[['gyro_x', 'gyro_y', 'gyro_z']]
    
    spectro_plot(acce_data, 100, 'Spectrogram of Accelerometer Data')
    spectro_plot(gyro_data, 100, 'Spectrogram of Gyroscopic Data')
    