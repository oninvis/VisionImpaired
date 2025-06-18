import source
from ultralytics import YOLO
main_model = YOLO('yolov8n.pt')
second_model = YOLO('potholedetector.pt')

source.detect_both_video(main_model,second_model)