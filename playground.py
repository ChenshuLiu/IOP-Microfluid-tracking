import cv2
import tkinter
import os
import pandas as pd
import numpy as np

# draw bounding box around the liquid compartment region
# give some point labels to define the liquid surface
# Lucas Kanade method to track the location of the liquid surface (relative to the label coordinates in step 2)

cap = cv2.VideoCapture('Demo_vid.mp4')
ret, frame = cap.read()
bbox = cv2.selectROI("Choose microfluidic chamber", frame, fromCenter = False, showCrosshair = True)
cv2.destroyWindow("Choose microfluidic chamber")
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)
while True:
    ret, frame = cap.read() # already RGB video format
    if not ret:
        break
    chamber_tracker, bbox = tracker.update(frame)
    if chamber_tracker:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        cv2.putText(frame, "Tracking microfluidic chamber", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed: cannot see chamber", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("IOP vid", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)