import torch
import cv2
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print(cv2.getBuildInformation())

# sudo apt update
# sudo apt install -y \
#   python3-gi \
#   gir1.2-gstreamer-1.0 \
#   gir1.2-gst-plugins-base-1.0 \
#   gstreamer1.0-tools \
#   gstreamer1.0-plugins-good \
#   gstreamer1.0-plugins-bad
