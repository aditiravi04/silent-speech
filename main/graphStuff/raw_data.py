import pandas as pd
import matplotlib.pyplot as plt

def convert_to_numeric(data):
    # str -> numeric (invalid data = get rid of)
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    return data

def plot_raw_data(csv_file):
    
    data = pd.read_csv(csv_file)
    
    # str->numeric, drop invalid 
    data = convert_to_numeric(data)
    data.dropna(inplace=True)  
    
    # accel
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)  
    
    plt.xlabel('Time')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Raw Accelerometer Data')
    
    plt.plot(data.index, data['acce_x'], label='Acceleration X', color='r')
    plt.plot(data.index, data['acce_y'], label='Acceleration Y', color='g')
    plt.plot(data.index, data['acce_z'], label='Acceleration Z', color='b')
    
    plt.grid()

    # gyro
    plt.subplot(2, 1, 2)  
    
    plt.xlabel('Time')
    plt.ylabel('Gyroscope (degrees/s)')
    plt.title('Raw Gyroscope Data')
    
    plt.plot(data.index, data['gyro_x'], label='Gyroscope X', color='r')
    plt.plot(data.index, data['gyro_y'], label='Gyroscope Y', color='g')
    plt.plot(data.index, data['gyro_z'], label='Gyroscope Z', color='b')

    plt.grid()

    plt.tight_layout()  # fit properly
    plt.show()


if __name__ == '__main__':
    plot_raw_data('left/left_03.csv')  # replace w/ whichever file to plot
