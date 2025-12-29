#!/usr/bin/env python3
"""
CSC 485 Final Project - FINAL VERSION
Author: Trevor Kkaaya
"""

import cv2
import numpy as np
import time
import os
import logging
import picar
from picar import back_wheels, front_wheels


# LOGGING SETUP
# Saves history of detections + optional debug images

if not os.path.exists("debug_logs"):
    os.makedirs("debug_logs")

logging.basicConfig(
    filename="track_history.log",
    level=logging.INFO,
    format="%(asctime)s - %(message)s"
)

def debug_image(name, frame):
    """Saves a debug image into /debug_logs"""
    filename = f"debug_logs/{name}_{time.time()}.jpg"
    cv2.imwrite(filename, frame)

#car hardware setup

DB_FILE = "/home/pi/SunFounder_PiCar-V/remote_control/remote_control/driver/config"

picar.setup()

# Steering + motor controllers
fw = front_wheels.Front_Wheels(debug=False, db=DB_FILE)
bw = back_wheels.Back_Wheels(debug=False, db=DB_FILE)

fw.ready()
bw.ready()
fw.turn_straight()

# DRIVING CONSTANTS
CENTER = 90                  #straight wheel direction
FORWARD_SPEED = 28           #normal driving speed
TURN_SPEED = 22              #slow speed for tight turns
KP = 0.70                    #steering proportional control
MAX_TURN = 45                #max left/right turn
STOP_WAIT = 3.0              #stop sign wait duration

bw.speed = FORWARD_SPEED     #start motors at normal speed

#state flags (for tracking stop mode, fail-safe, turning state)

line_missing = False         #yellow line temporarily lost
turn_mode = False            #car is in middle-of-turn correction
stop_approach = False        #car is approaching stop sign
stop_zone_override = False   #disables fail-safe when stop bar is ahead
last_seen_time = time.time() #last time the yellow line was seen



#COLOR MASKS — DETECT YELLOW LINE + RED STOP BAR / LIGHT

def yellow_mask(frame):
    """
    Detect yellow pixels using HSV thresholds.
    Works reliably under classroom lighting.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([18, 60, 80])
    upper = np.array([45, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3)))
    mask = cv2.morphologyEx(mask, np.ones((5,5)), cv2.MORPH_CLOSE)
    return mask


def red_mask(frame):
    """
    Detect red stop sign or traffic light using HSV.
    Handles the two red hue ranges automatically.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red appears in two hue clusters
    lower1 = np.array([0, 80, 80])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([170, 80, 80])
    upper2 = np.array([179, 255, 255])

    return cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)



def get_centroid(mask):
    """Extract the x-coordinate of a detected line region."""
    M = cv2.moments(mask)
    if M["m00"] > 1e-3:
        return int(M["m10"] / M["m00"])
    return None



#LINE FOLLOWING + TURNING LOGIC 

def get_line_steering_angle(frame):
    global line_missing, last_seen_time, turn_mode, stop_zone_override

    h, w = frame.shape[:2]

    #bottom ROI — main line following
    roi_bottom = frame[int(h*0.65):h, :]

    #mid ROI — detects turns earlier (before bottom ROI sees shift)
    roi_mid = frame[int(h*0.45):int(h*0.65), :]

    mask_b = yellow_mask(roi_bottom)
    mask_m = yellow_mask(roi_mid)

    cx_b = get_centroid(mask_b)
    cx_m = get_centroid(mask_m)
    
   
    # If the yellow line appears heavily offset in mid-ROI,
    # the robot begins turning early.

    if cx_m is not None:
        offset = (cx_m - w/2) / (w/2)

        if abs(offset) > 0.35:
            turn_mode = True
            bw.speed = TURN_SPEED   # slow down for cornering
            angle = CENTER + int(offset * 60)
            return max(CENTER - MAX_TURN, min(CENTER + MAX_TURN, angle)), frame

    # Exit turn mode when the line becomes centered again
    if turn_mode and cx_m is not None and abs((cx_m - w/2)/(w/2)) < 0.20:
        turn_mode = False
        bw.speed = FORWARD_SPEED


    # FAIL-SAFE (only active when stop bar is NOT detected)
    # Prevents the car from driving blindly if yellow line is lost.
    if (not stop_zone_override) and cx_b is None and cx_m is None:

        if not line_missing:
            print("Fail-safe activated")
            line_missing = True
            last_seen_time = time.time()

        bw.stop()
        elapsed = time.time() - last_seen_time

        #Try sweeping right
        if elapsed < 0.5:
            fw.turn(CENTER + 40)
            return CENTER + 40, frame

        #Then sweep left
        elif elapsed < 1.0:
            fw.turn(CENTER - 40)
            return CENTER - 40, frame

        #or else wait straight
        else:
            fw.turn(CENTER)
            return CENTER, frame

    # Reset fail-safe flag
    line_missing = False


    # NORMAL LINE FOLLOWING
    if cx_b is None:
        return CENTER, frame  # failsafe fallback

    bw.speed = FORWARD_SPEED

    error = (cx_b - w/2) / (w/2)
    angle = CENTER + int(KP * error * 45)

    return max(CENTER - MAX_TURN, min(CENTER + MAX_TURN, angle)), frame



#STOP SIGN DETECTION (EARLY + FINAL)
def detect_stop_bar(frame):
    """
    early_detect  = red detected ahead (prepare)
    final_detect  = red detected at bottom (STOP NOW)
    """
    h, w = frame.shape[:2]

    # Early detection — red appearing in the lower-middle region
    early_roi = frame[int(h * 0.75):int(h * 0.92), :]
    early_mask = red_mask(early_roi)
    early_mask = cv2.morphologyEx(early_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    early_detect = cv2.countNonZero(early_mask) > 700

    # Final stop detection — car is right on top of the stop tape
    final_roi = frame[int(h * 0.90):h, :]
    final_mask = red_mask(final_roi)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))
    final_detect = cv2.countNonZero(final_mask) > 800

    return early_detect, final_detect, frame




#TRAFFIC LIGHT DETECTION
def detect_traffic_light(frame):
    """Detect red or green traffic light at the top center."""
    h, w = frame.shape[:2]
    roi = frame[:int(h*0.40), int(w*0.25):int(w*0.75)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0,120,90]), np.array([10,255,255]))
    red2 = cv2.inRange(hsv, np.array([170,120,90]), np.array([179,255,255]))

    green_mask = cv2.inRange(hsv, np.array([40,70,70]), np.array([90,255,255]))

    if cv2.countNonZero(red1 | red2) > 400:
        return "red", frame

    if cv2.countNonZero(green_mask) > 400:
        return "green", frame

    return "none", frame


# MAIN CONTROL LOOP

def main():
    global stop_zone_override, stop_approach

    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)

    if not cap.isOpened():
        print("Camera error.")
        return

    print("Autonomous mode started.")
    last_stop = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            continue

        angle, frame = get_line_steering_angle(frame)
        early_stop, final_stop, frame = detect_stop_bar(frame)
        light_state, frame = detect_traffic_light(frame)

        now = time.time()

      
        #disables fail-safe when red stop bar seen
        stop_zone_override = early_stop or final_stop

        #slow down before final stop
        if early_stop and not final_stop and (now - last_stop) > 4:
            stop_approach = True
            bw.speed = 23
            fw.turn_straight()
            print("Approaching stop bar...")


       
        #directly on top of red tape
        if final_stop and (now - last_stop) > 4:
            print("STOP — waiting 3 seconds…")
            stop_zone_override = False
            stop_approach = False
            bw.stop()
            fw.turn_straight()

            last_stop = now
            time.sleep(STOP_WAIT)

        #red means STOP
        elif light_state == "red":
            bw.stop()
            fw.turn_straight()


        #normal driving 
        elif not stop_approach:
            bw.speed = FORWARD_SPEED
            bw.forward()
            fw.turn(angle)


        #display camera feed
        cv2.imshow("Final Project View", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    bw.stop()
    fw.turn_straight()
    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()