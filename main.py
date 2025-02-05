import cv2
import tkinter
import os
import pandas as pd
import numpy as np

# draw bounding box around the liquid compartment region
# give some point labels to define the liquid surface
# Lucas Kanade method to track the location of the liquid surface (relative to the label coordinates in step 2)

cap = cv2.VideoCapture('Demo_vid.mp4')
ret, frame = cap.read() # labeling (bbox and liquid surface) on the very first frame
# clicking event for selecting the points to define the liquid surface
select_liquid_surface = False # won't start the selection unless after the bbox definition
liquid_surface_define_points = []
def define_liquid_surface(event, x, y, flags, param):
    global liquid_surface_define_points, select_liquid_surface
    if event == cv2.EVENT_LBUTTONDOWN and select_liquid_surface:
        liquid_surface_define_points.append((x, y))

bbox = cv2.selectROI("Choose microfluidic chamber", frame, fromCenter = False, showCrosshair = True) # draw bounding box
cv2.destroyWindow("Choose microfluidic chamber")
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

lk_params = dict(winSize = (5, 5),
                 maxLevel = 2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))

# after initializing the bbox tracker, allow user to define the liquid surface
select_liquid_surface = True
cv2.namedWindow("Define Liquid Surface")
cv2.setMouseCallback("Define Liquid Surface", define_liquid_surface)
while True: # outline the fluid surface by pointing
    frame_display = frame.copy()
    # Draw the selected bounding box
    x, y, w, h = map(int, bbox)
    cv2.rectangle(frame_display, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Draw selected points
    for point in liquid_surface_define_points:
        cv2.circle(frame_display, point, 1, (255, 0, 0), -1)

    cv2.imshow("Define Liquid Surface", frame_display)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Press Enter to finish selecting points
        break
cv2.destroyWindow("Define Liquid Surface")  # Close point selection window
print(liquid_surface_define_points) # the outlined liquid surface coordinates are stored here

old_frame = None
old_points = None
new_points = []

# tracking loop
while True:
    # print()
    ret, frame = cap.read() # already RGB video format
    if not ret:
        break
    chamber_tracker, bbox = tracker.update(frame) # keep focus on the liquid chamber
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # lucas kanade method is for intensity comparison, reduce to single channel for easy operation
    if chamber_tracker: # when the chamber is clearly shown
        if old_frame is None: # first frame of surface tracking
            old_frame = frame_grayscale.copy()
            old_points = np.array(liquid_surface_define_points, dtype = np.float32).reshape(-1, 1, 2)
        new_points, tracking_status, _ = cv2.calcOpticalFlowPyrLK(old_frame, frame_grayscale, old_points, None, **lk_params)
        # print("old points:", old_points)
        # print("new points:", new_points)

        # show the bounding box of liquid chamber
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2, 1)
        cv2.putText(frame, "Tracking microfluidic chamber", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # show the liquid surface trackings
        print("entering point display for loop")
        for i, (new, old) in enumerate(zip(new_points, old_points)):
            new_x, new_y = new.ravel()
            old_x, old_y = old.ravel()
            cv2.circle(frame, (int(new_x), int(new_y)), 5, (0, 255, 0), -1)
            cv2.line(frame, (int(new_x), int(new_y)), (int(old_x), int(old_y)), (0, 255, 0), 2)

        # replace the old frame and points with new ones for next frame checking
        old_frame = frame_grayscale.copy()
        old_points = new_points.reshape(-1, 1, 2) # with shape (point_idx, 1 row, 2 dimensions)
        # print("old points:", old_points)
    else:
        cv2.putText(frame, "Tracking failed: cannot see chamber", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("IOP vid", frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # press q to quit
        break

cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)