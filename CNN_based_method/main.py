from helper import CNN_Model, RollingBuffer, red_region_crop, DynamicViT
import torch
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as transforms

device = torch.device("mps" if torch.cuda.is_available() else "cpu")

# pt file, containing state dict, optimizer information, epoch, stored as dictionary
checkpoints = torch.load("model_checkpoints/checkpoint_epoch_50_red_enhanced.pt", map_location=device)
model_checkpoint = checkpoints['model_state_dict']
model = CNN_Model().to(device)
model.load_state_dict(model_checkpoint)
model.eval()

# pth file, only the state dict
# model = CNN_Model().to(device)
# state_dict = torch.load("cnn_model_20ep.pth")
# model.load_state_dict(state_dict)
# model.eval()

# checkpoints = torch.load("model_checkpoints/checkpoint_epoch_24.pt", map_location=device)
# model_checkpoint = checkpoints['model_state_dict']
# model = DynamicViT()
# model.load_state_dict(model_checkpoint)
# model.to(device)
# model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),  # Keep original dimensions
])

# video processing pipeline
cap = cv2.VideoCapture("../Demo_vid.mp4")
ret, frame = cap.read()
bbox = cv2.selectROI("Choose microfluidic chamber region",
                     frame, fromCenter = False,
                     showCrosshair = True)
cv2.destroyWindow("Choose microfluidic chamber region")
tracker = cv2.TrackerKCF_create()
tracker.init(frame, bbox)

pressure_buffer = RollingBuffer(10) # for averaging the pressure interpreted to make the reading smoother

while True:
    ret, frame = cap.read()
    if not ret:
        break
    chamber_tracker, bbox = tracker.update(frame)
    if chamber_tracker:
        x, y, w, h = map(int, bbox)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)
        cv2.putText(frame, "Tracking microfluidic chamber", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
        ROI_region = frame[y:y+h, x:x+w] # only analyze the selected region (numpy array)

        ROI_region_red_only = red_region_crop(ROI_region, effect = "enhance")

        ROI_pil = Image.fromarray(ROI_region_red_only)
        ROI_tensor = transform(ROI_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_pressure = model(ROI_tensor).item()
            pressure_buffer.add(predicted_pressure)
        cv2.putText(frame, f"Pressure: {np.mean(pressure_buffer.get()):.1f}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failed: cannot detect the microfluidic region selected",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("IOP tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()