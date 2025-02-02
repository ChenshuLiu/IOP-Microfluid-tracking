import cv2
import tkinter
import os
import pandas as pd
import numpy as np

# draw bounding box around the liquid compartment region
# give some point labels to define the liquid surface
# Lucas Kanade method to track the location of the liquid surface (relative to the label coordinates in step 2)
