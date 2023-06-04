from PIL import Image, ImageDraw
import cv2 as cv

from predict import KBDPredictor
from utils import STATE_DICT_PATH

def main():
  predictor = KBDPredictor(STATE_DICT_PATH, threshold=0.4)

  cap = cv.VideoCapture(1)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
  
    img = Image.fromarray(cv_to_pil(frame))
    # img = Image.open("../butters-keyboard-1.jpg")

    result, cat_bbs, kb_bbs = predictor.predict(img)

    print(result)

    img_bb = img.copy()
    img_draw_bb = ImageDraw.Draw(img_bb)

    for bbox in cat_bbs:
      img_draw_bb.rectangle(bbox.tolist(), outline='white')
    
    for bbox in kb_bbs:
      img_draw_bb.rectangle(bbox.tolist(), outline='red')

    # img_bb.show()
    cv.imshow('frame', pil_to_cv(img))

def pil_to_cv(img: Image.Image) -> cv.Mat:
  return cv.cvtColor(img, cv.COLOR_RGB2BGR)

def cv_to_pil(img: cv.Mat) -> Image.Image:
  return cv.cvtColor(img, cv.COLOR_BGR2RGB)

if __name__ == "__main__":
  main()