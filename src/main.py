import torch
from PIL import Image, ImageDraw, ImageFont
import cv2 as cv
import numpy as np
import time
import threading
import tkinter as tk
from tkinter import messagebox

from predict import KBDPredictor
from utils import *

MAX_FPS = 30
FILTER_SECONDS = 0.5

result_g = False
result_lock = threading.Lock()

def main():
  predictor = KBDPredictor(None)

  cap = cv.VideoCapture(0)
  if not cap.isOpened():
    print("Cannot open camera")
    exit()
  
  prev_frame_time = 0
  new_frame_time = 0

  # moving average filter, newest at front
  result_filter = [0] * int(MAX_FPS * FILTER_SECONDS)

  alert_thread = threading.Thread(target=alert_thread_fn, daemon=True)
  alert_thread.start()

  while True:
    ret, frame = cap.read()
    if not ret:
      print("Can't receive frame (stream end?). Exiting ...")
      break
  
    img = cv_to_pil(cv.flip(frame, 1))

    result, cat_bbs, kb_bbs = predictor.predict(img)

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # update filter
    result_filter = [int(result)] + result_filter[:-1]
    wnd_end = max(int(min(fps, MAX_FPS) * FILTER_SECONDS), 1)
    wnd = result_filter[:wnd_end]
    mavg = sum(wnd) / len(wnd)

    with result_lock:
      global result_g
      result_g = mavg > 0.5

    ann_img = Image.fromarray(img)
    annotate_frame(ann_img, result, cat_bbs, kb_bbs, fps)

    cv.imshow('Keyboard Defender', pil_to_cv(ann_img))
    if cv.waitKey(1) == ord('q'):
      break
  
  cap.release()
  cv.destroyAllWindows()

def alert_thread_fn():
  root = tk.Tk()
  root.attributes("-topmost", True)
  root.withdraw()
  while True:
    with result_lock:
      result = result_g
    if result:
      # without this, sometimes warning won't show
      root.wm_deiconify()
      root.focus_force()
      messagebox.showwarning(
        'Keyboard Defender Alert',
        'A cat is attacking your keyboard!'
      )
      root.withdraw()
    else:
      time.sleep(0.1)

def annotate_frame(img: Image.Image,
                   result: bool,
                   cat_bbs: torch.Tensor,
                   kb_bbs: torch.tensor,
                   fps: float):
  draw_bbs(img, cat_bbs, kb_bbs)
  draw_msg = ImageDraw.Draw(img)
  if result:
    draw_msg.text((0, 0), "DANGER", font=ImageFont.truetype("arial.ttf", 20), fill='red')
  else:
    draw_msg.text((0, 0), "SAFE", font=ImageFont.truetype("arial.ttf", 20), fill='lime')

  draw_msg.text((0, 24), f'{fps:.2f}', font=ImageFont.truetype("arial.ttf", 20), fill='lime')

def draw_bbs(img: Image.Image, cat_bbs: torch.Tensor, kb_bbs: torch.tensor) -> Image.Image:
  img_draw_bb = ImageDraw.Draw(img)

  for bbox in cat_bbs:
    img_draw_bb.rectangle(bbox.tolist(), outline='white')
  
  for bbox in kb_bbs:
    img_draw_bb.rectangle(bbox.tolist(), outline='red')

def pil_to_cv(img: np.ndarray) -> cv.Mat:
  return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

def cv_to_pil(img: cv.Mat) -> np.ndarray:
  return np.array(cv.cvtColor(img, cv.COLOR_BGR2RGB))

if __name__ == "__main__":
  main()