import torch
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
import numpy as np

from predict import KBDPredictor
from utils import STATE_DICT_PATH

def main():
  predictor = KBDPredictor(STATE_DICT_PATH)

  cap = cv.VideoCapture(0)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
  
    img = cv_to_pil(cv.flip(frame, 1))
    # img = Image.open("../butters-keyboard-1.jpg")

    result, cat_bbs, kb_bbs = predictor.predict(img)

    img_bb = draw_bbs(img, cat_bbs, kb_bbs)
    draw_msg = ImageDraw.Draw(img_bb)
    if result:
      draw_msg.text((0, 0), "DANGER", font=ImageFont.truetype("arial.ttf", 20), fill='red')
    else:
      draw_msg.text((0, 0), "SAFE", font=ImageFont.truetype("arial.ttf", 20), fill='lime')
  
    # img_bb.show()
    cv.imshow('frame', pil_to_cv(img_bb))
    if cv.waitKey(1) == ord('q'):
      break
  
  cap.release()
  cv.destroyAllWindows()

def draw_bbs(img: np.ndarray, cat_bbs: torch.Tensor, kb_bbs: torch.tensor) -> Image.Image:
  img_bb = Image.fromarray(img)
  img_draw_bb = ImageDraw.Draw(img_bb)

  for bbox in cat_bbs:
    img_draw_bb.rectangle(bbox.tolist(), outline='white')
  
  for bbox in kb_bbs:
    img_draw_bb.rectangle(bbox.tolist(), outline='red')
  
  return img_bb

def pil_to_cv(img: np.ndarray) -> cv.Mat:
  return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

def cv_to_pil(img: cv.Mat) -> np.ndarray:
  return np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

if __name__ == "__main__":
  main()