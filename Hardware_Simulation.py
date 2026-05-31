import serial
import random
import time

SERIAL_PORT = 'COM10' 
BAUD_RATE = 115200

try:
    # Initialize Serial connection
    arduino = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # Allow connection to stabilize
    print("Smart Grid RL Simulation")
    print("========================================\n")

    while True:
        # Generate a random hour for the scenario
        hour = random.randint(0, 23)
        time_str = f"{hour:02d}:00"
        
        print(f"[Scenario -> Hour {time_str}]")

        # --- Scenario Logic & Hardware Triggering ---
        if 10 <= hour <= 16:
            print("  State: Running on Solar Power")
            print(" Hardware: LED is [SOLID ON]")
            arduino.write(b'S') # Trigger Arduino Solar state
            reward = 2.50
            print(f" RL Reward: +${reward:.2f} (Free energy utilized)")
            
        elif 17 <= hour <= 21:
            print(" State: Discharging Home Battery")
            print(" Hardware: LED is [BLINKING]")
            arduino.write(b'B') # Trigger Arduino Battery state
            reward = 4.80
            print(f" RL Reward: +${reward:.2f} (Avoided expensive peak rates!)")
            
        else:
            print(" State: Drawing from Grid")
            print(" Hardware: LED is [OFF]")
            arduino.write(b'G') # Trigger Arduino Grid state
            reward = -1.20
            print(f" RL Reward: -${abs(reward):.2f} (Paid off-peak grid rate)")

        print("-" * 40)
        time.sleep(3) # Pause for 3 seconds between scenarios

except serial.SerialException:
    print(f"Error: Could not connect to {SERIAL_PORT}. Check connection and close Arduino Serial Monitor.")
except KeyboardInterrupt:
    print("\nSimulation stopped by user.")
finally:
    if 'arduino' in locals() and arduino.is_open:
        arduino.close()