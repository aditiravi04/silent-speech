import os
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Function to load dataset
def load_data(label_df, data_dir):
    # Empty lists to store features and labels
    features = []
    labels = []

    for _, row in label_df.iterrows():
        filename = os.path.join(data_dir, row['filename'] + ".csv")

        # Read file into pandas dataframe
        df = pd.read_csv(filename)

        # Keep only accelerometer and gyroscope signals
        data = df[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].values.astype(np.float32)

        # Normalize data
        data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

        # Populate lists with normalized data and labels
        features.append(data.flatten())
        labels.append(row['0'])

    return np.array(features), np.array(labels)

def load_data2(data_dir):
    features = []
    labels = []

    for file_name in os.listdir(data_dir):
        label = file_name.split('_')[0] 
        file_path = os.path.join(data_dir, file_name)
        df = pd.read_csv(file_path)
            
        # Keep only accelerometer and gyroscope signals
        data = df[['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']].values.astype(np.float16)

        if len(data) < 300:
            data = np.pad(data, ((0, 300-len(data)), (0,0)))
        data = data[:300]
        
        # Normalize data
        data_min = np.min(data, axis=0)
        data_max = np.max(data, axis=0)
        data = (data - data_min) / (data_max - data_min)
        # Populate lists with normalized data and labels
        features.append(data.flatten())
        labels.append(label)
    
    return np.array(features), np.array(labels)
    

def train_and_evaluate_svm(X_train, y_train, X_test, y_test):
    # Create the SVM classifier
    svm_classifier = SVC(kernel='rbf')

    # Train the classifier
    svm_classifier.fit(X_train, y_train)

    joblib.dump(svm_classifier, 'svm_classifier.pkl')

    # Perform prediction on the test set
    y_pred = svm_classifier.predict(X_test)
    counts = {"up":0, "down":0, "right":0, "left":0}
    for i in y_test:
        counts[i] += 1
    print(counts)
    counts = {"up":0, "down":0, "right":0, "left":0}
    for i in y_pred:
        counts[i] += 1
    print(counts)
   
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM accuracy: {accuracy:.3%}')

    # Plot the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, cmap="Blues", xticklabels=["down", "left", "right", "up"], yticklabels=["down", "left", "right", "up"])
    plt.title('confusion matrix')
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.show()

X, y = load_data2("data")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform training and testing with SVM
train_and_evaluate_svm(X_train, y_train, X_test, y_test)

# Assignment 3
import serial

import time
from collections import deque

SERIAL_PORT = 'COM8'
BUAD_RATE = 115200

def send_to_board(ser, command):
    ser.write((command + '\n').encode('utf-8'))
    print(f"sent command: {command}")

def main_assignment3():

    # Perform training and testing with SVM
    ser = serial.Serial(SERIAL_PORT, BUAD_RATE, timeout = 1)
    
    try:
        print(f"connected to {SERIAL_PORT} :)")
        time.sleep(2)

        send_to_board(ser, "START")
        #read_and_save(ser, file, 4)
        real_time_prediction(ser, 60)

        send_to_board(ser, "STOP")

    except Exception as e:
        print(f"ERROR: {e}")
    
    finally:
        if ser.is_open:
            ser.close()
            print("Serial port closed.")

def real_time_prediction(ser, duration=10):
    start_time = time.time()

    buffer_size = 300  # Number of readings to match 360 features
    feature_window = deque(maxlen=buffer_size)

    svm_classifier = joblib.load('svm_classifier.pkl')

    print("Starting real-time prediction. Press Ctrl+C to stop.")

    try:
        while ((time.time() - start_time) < duration):
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8').strip()
                    #print(f"LINE: {line}")

                    data = [float(x) for x in line.split()]
                    if len(data) == 6:  # Ensure all  values are present
                        feature_window.append(data)

                        if len(feature_window) == buffer_size:
                            # Flatten the sliding window into a single feature vector
                            window_data = np.array(feature_window, dtype=np.float16)
                            data_min = np.min(window_data, axis=0)
                            data_max = np.max(window_data, axis=0)
                            data_range = np.where((data_max - data_min) == 0, 1, data_max - data_min)
                            normalized_data = (window_data - data_min) / data_range

                            # Flatten the normalized data
                            input_features = normalized_data.flatten().reshape(1, -1)
                            
                            # Perform gesture prediction
                            prediction = svm_classifier.predict(input_features)
                            print(f"This is the predicted word: {prediction[0]}. Say your next word now...")
                            feature_window = deque(maxlen=buffer_size)
                            
                except Exception as e:
                    print(f"ERROR: {e}")
                    break
    except KeyboardInterrupt:
        print("Stopped real-time prediction.")

main_assignment3()