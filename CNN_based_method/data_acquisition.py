'''
For extracting frames from video
'''

import cv2
import os

# Function to extract and save each frame from a video
def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        output_full_file_folder = os.path.join(output_folder, "full_frame")
        os.makedirs(output_full_file_folder)
        ROI_file_folder = os.path.join(output_folder, "ROI")
        os.makedirs(ROI_file_folder)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # select the region of the microfluidic regions for classification
    ret, frame = cap.read()
    bbox = cv2.selectROI("Choose microfluidic chamber",
                         frame, fromCenter = False,
                         showCrosshair = True)
    cv2.destroyWindow("Choose microfluidic chamber")
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, bbox)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        
        # If the frame was read successfully
        if ret:
            frame_count += 1
            
            chamber_tracker, bbox = tracker.update(frame)
            if chamber_tracker:
                x, y, w, h = map(int, bbox)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)
                cv2.putText(frame, "Tracking microfluidic chamber", 
                            (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.6, (0, 255, 0), 2)
                ROI_frame = frame[y:y+h, x:x+w]
                ROI_filename = os.path.join(ROI_file_folder, f"frame_{frame_count:03d}_ROI.jpg")
                print(ROI_filename)
                cv2.imwrite(ROI_filename, ROI_frame)
            else:
                cv2.putText(frame, "Tracking failed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("IOP vid", frame)
            # Save the frame to the output folder
            frame_filename = os.path.join(output_full_file_folder, f"frame_{frame_count:03d}.jpg")
            cv2.imwrite(frame_filename, frame) # write whole frame
            
        else:
            # End of video
            break
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    
    # Release the video capture object
    cap.release()
    # Close any open OpenCV windows (if any)
    cv2.destroyAllWindows()
    
    print(f"Extracted {frame_count} frames to {output_folder}")

# Example usage
video_path = 'Demo_vid.mp4'  # Path to the input video file
output_folder = 'CNN_frame_data'  # Folder to save the extracted frames

# Extract frames from the video
extract_frames(video_path, output_folder)
