import os
import pandas as pd
import serial
import time
import sys
import threading
import pickle
import numpy as np
from filelock import FileLock

def process(csv_file):
    df = pd.read_csv(csv_file)
    df[['gyro_x',  'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']] = df[['gyro_x',  'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].apply(pd.to_numeric, errors='coerce')

    df.fillna(0, inplace=True)

    data = df[['gyro_x',  'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].values.astype(np.float32)
    if len(data) < 400:
        data = np.pad(data, ((0, 400-len(data)), (0,0)))
    data = data[:400]
    
    # Normalize data
    data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    
    return data.flatten()

def classify_movement():
    with open('svm_classifier.pkl', 'rb') as f:
        svm_classifier = pickle.load(f)
        
    # with open('knn_classifier.pkl', 'rb') as f:
    #     knn_classifier = pickle.load(f)

    file = "curr.csv"
    lock = FileLock(f"{file}.lock")
    
    while True:
        if os.path.exists(file) and os.path.getsize(file) > 0:

            try:
                curr_data = process(file)
                curr_data = curr_data.reshape(1, -1)
                
                predict_with_svm = svm_classifier.predict(curr_data)[0]
                #predict_with_knn = knn_classifier.predict(curr_data)[0]
                
                print(f"SVM Prediction: {predict_with_svm}")
                #print(f"KNN Prediction: {predict_with_knn}")
                print("\n")
            
            except Exception as e:
                print(f"There's an error while trying to classify this motion: {e}")
        else:
            print("Waiting for there to be info in file")
        
        time.sleep(4)
            

def main():
    port = 'COM8'
    
    try:
        ser = serial.Serial(port, baudrate = 115200, timeout = 1)
        print("Connected")
    except serial.SerialException as e:
        print(f"Error {e}")
        sys.exit(1)


    
    dataList = []  # List to store the extracted values
    df = pd.DataFrame(dataList, columns=["gyro_x",  "gyro_y", "gyro_z", "acce_x", "acce_y", "acce_z"])  # Create a DataFrame with the extracted values
    
    #print("\ngot here\n")
    
    csv_file = 'curr.csv'
    lock = FileLock(f"{csv_file}.lock")
    start_time = time.time()
    try:
        classify_now = threading.Thread(target=classify_movement, daemon=True)
        classify_now.start()
        
        # start_time = time.time()
        # duration = 4
        # while time.time() - start_time < duration:
        while True:
            #print("while true\n")
            try:
                if ser.in_waiting > 0:
                    #print("\ninside if")
                    data = ser.readline().decode().strip()
                    
                    #Filter - Remove unneccary print
                    
                    #Split by space
                    row = data.split()
                    
                    if len(row) == 6:
                        df.loc[len(df)] = row

                    if time.time() - start_time >=4:
                        print(f"Start the next movement now. I will print what movement you did previously here: ")
                        with lock:
                            df.to_csv(csv_file, mode='w', header=True, index=False)
                        df = pd.DataFrame(columns=["gyro_x", "gyro_y", "gyro_z", "acce_x", "acce_y", "acce_z"])
                        start_time = time.time()
                    
            except Exception as e:
                print(f"After True{e}")

    except KeyboardInterrupt:
        print("Oh no")
        
    finally:
        ser.close()
        print("Connection closed")

if __name__ == '__main__':
    main()