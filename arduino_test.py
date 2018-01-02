import serial
import time

arduino = serial.Serial("/dev/ttyUSB0", 9600, timeout=.1)
time.sleep(1)

print("all stop")
arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
time.sleep(0.5)
input("Press Enter to continue...")

print("forward and backwards")
arduino.write(("<" + str(4) + "," + str(4) + ">").encode())
time.sleep(2)

arduino.write(("<" + str(-4) + "," + str(-4) + ">").encode())
time.sleep(2)

print("all stop")
arduino.write(("<" + str(0) + "," + str(0) + ">").encode())
