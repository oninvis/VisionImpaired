# ---------------------------------------------------------------------------------------------------------
# Importing all the libraries
# ---------------------------------------------------------------------------------------------------------
import cv2 
import numpy as np 
import pyttsx3
# ---------------------------------------------------------------------------------------------------------
# Ensembling
# ---------------------------------------------------------------------------------------------------------
def combine_detections(r1, r2 , main_model , second_model):
    '''
    r1 is the results from the main model and r2 is the results from second model
    '''
    # extract boxes and scores as python lists 
    boxes = r1.boxes.xyxy.cpu().numpy().tolist() + r2.boxes.xyxy.cpu().numpy().tolist()
    scores = r1.boxes.conf.cpu().numpy().tolist() + r2.boxes.conf.cpu().numpy().tolist()
    # maps label names for each of the models
    labels_main = [main_model.names[int(c)] for c in r1.boxes.cls]
    labels_second = [second_model.names[int(c)] for c in r2.boxes.cls]
    labels = labels_main + labels_second

    return boxes , scores , labels 
# ---------------------------------------------------------------------------------------------------------
def draw_detections(frame , boxes , scores , labels , box_color = (0,255,0) , text_color=(0,255,0)):
    """
    Draws bounding boxes and labels onto `frame`.
    Expects:
      - boxes: list of [x1, y1, x2, y2]
      - scores: list of floats
      - labels: list of strings
    """
    for (x1,y1,x2,y2) , score , lbl in zip(boxes,scores,labels):
        #converting pixels in co-ordinates 
        x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))

        # draw a bounding box 
        cv2.rectangle(frame,(x1,y1) ,(x2,y2),box_color,2)
        # draw label and confidence
        text = f"{lbl} {score:.2f}"
        cv2.putText(frame , text,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX , 0.5 , text_color , 1)
# ---------------------------------------------------------------------------------------------------------
# Approximate distance calculation 
# ---------------------------------------------------------------------------------------------------------
def get_nearest_by_box_area(boxes, labels, scores):
    """
    Picks the detection with the largest pixel area ⇒ assumed closest.
    Returns dict with index, box, label, score, area.
    """
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes]
    idx   = int(np.argmax(areas)) if areas else None
    if idx is None:
        return None
    return {
        "index": idx,
        "box":   boxes[idx],
        "label": labels[idx],
        "score": scores[idx],
        "area":  areas[idx]
    }
# ---------------------------------------------------------------------------------------------------------
def describe_proximity(nearest, frame_shape):
    """
    Converts nearest["area"] into a qualitative range and side cue.
    Returns a string like "A person on your right, very close" or None.
    """
    frame_h, frame_w = frame_shape[:2]
    frame_area = frame_h * frame_w
    ratio = nearest["area"] / (frame_area + 1e-6)

    # tune these thresholds as needed
    if   ratio > 0.15:  dist_desc = "very close"
    elif ratio > 0.05:  dist_desc = "close"
    elif ratio > 0.01:  dist_desc = "mid-range"
    else:               return None   # too far

    x1,_,x2,_ = nearest["box"]
    cx = (x1 + x2) / 2
    if   cx < frame_w/3:        side = "left"
    elif cx > 2*frame_w/3:      side = "right"
    else:                       side = "ahead"

    return f"A {nearest['label']} on your {side}, {dist_desc}"
# ---------------------------------------------------------------------------------------------------------
# Speech To Text 
# ---------------------------------------------------------------------------------------------------------
_engine = None
def init_tts(rate: int = 150, volume: float = 1.0, voice_idx: int = 1):
    global _engine
    if _engine is None:
        # 1) initialize
        _engine = pyttsx3.init()
        # 2) pick your voice
        voices = _engine.getProperty('voices')            
        if 0 <= voice_idx < len(voices):
            _engine.setProperty('voice', voices[voice_idx].id)
        # 3) set rate & volume
        _engine.setProperty('rate', rate)
        _engine.setProperty('volume', volume)
# ---------------------------------------------------------------------------------------------------------
def speak(text: str):
    """Speak the given text (non-blocking)."""
    if _engine is None:
        init_tts()
    _engine.say(text)
    _engine.runAndWait()
# ---------------------------------------------------------------------------------------------------------
# Main module 
# ---------------------------------------------------------------------------------------------------------
def detect_both_video(main_model, second_model , source = 0 , conf = 0.25 , iou = 0.45 ):
    '''
    This runs detection on a video stream (0=default webcam ) and press 'q' to exit
    '''

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f'Error : Unable to open video source {source}')
        return 
    
    while True:
        ret , frame = cap.read()
        if not ret:
            break
        # running inference
        r1 = main_model(frame , conf=conf,iou = iou )[0]
        r2 = second_model(frame , conf=conf , iou = iou)[0]

        # combine and draw 
        boxes , scores , labels = combine_detections(r1 , r2,main_model,second_model)
        draw_detections(frame,boxes , scores , labels)

        # pick nearest by area & describe
        nearest = get_nearest_by_box_area(boxes, labels, scores)
        if nearest:
            desc = describe_proximity(nearest, frame.shape)
            # 4) highlight nearest in red
            x1,y1,x2,y2 = map(int, nearest["box"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            if desc:
                cv2.putText(frame,
                            desc,
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0,0,255), 2)
                print(desc)
                speak(desc)

        # display 
        cv2.imshow('live detections' ,frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()