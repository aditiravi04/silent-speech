import pandas as pd
import serial
import time
import sys

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
    
    csv_file = 'data/right_25.csv'
    try:
        start_time = time.time()
        duration = 3
        while time.time() - start_time < duration:
            #print("while true\n")
            try:
                if ser.in_waiting > 0:
                    #print("\ninside if")
                    data = ser.readline().decode().strip()
                    print("Recieved")
                    
                    #Filter - Remove unneccary print
                    
                    #Split by space
                    row = data.split()
                    
                    if len(row) == 6:
                        df.loc[len(df)] = row

                    df.to_csv(csv_file, index = False)
            except Exception as e:
                print(f"After True{e}")

    except KeyboardInterrupt:
        print("Oh no")
        
    finally:
        ser.close()
        print("Connection closed")

if __name__ == '__main__':
    main()