# VisionImpaired Helper
## Models
The program uses Ultralytics YOLO , yolov8n.pt , model. The other model ( potholedetector.pt ) which detects potholes is a finetuned version of yolov8n.pt. The program ensembles the original model by ultralytics with the fine tuned pothole detector model. The model was trained on over 3000 images of different potholes for different classes like manholes , potholes , pits etc. 
## Distance calculation
The program utilizes the camera for distance calculations, it uses the box area and frame's ratio as determining factor to detect whether the object is very close , close or mid-range to the user. It uses x co-ordinates of the boxes in order to determine whether the object is on the right , left or ahead of the user. 
## Speech to text 
The program uses the basic speech to text conversion method purely based on python which means it can work on any device ,and it uses default voice to concvert the text.
