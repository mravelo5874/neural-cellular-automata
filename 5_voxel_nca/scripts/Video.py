#@title Notebook Utilities and Setup

import base64
import io
import numpy as np
import os
import PIL.Image, PIL.ImageDraw, PIL.ImageFont
import requests
from IPython.display import Image

os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def imread(url, max_size=None, mode=None):
  if url.startswith(('http:', 'https:')):
    headers = {'User-Agent': 'Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0'}
    r = requests.get(url, headers=headers)
    f = io.BytesIO(r.content)
  else:
    f = url
  img = PIL.Image.open(f)
  if max_size is not None:
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  if mode is not None:
    img = img.convert(mode)
  img = np.float32(img) / 255.
  return img

def np2pil(a):
  if a.dtype in (np.float32, np.float64):
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None, quality=95):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=quality)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if a.ndim == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

def imshow(a, fmt='jpeg', scale=4):
  display(Image(data=imencode(zoom(a, scale), fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(a.shape[0])))
  th, tw = a.shape[1:3]
  pad = (w - a.shape[0]) % w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = a.shape[0] // w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

class VideoWriter:
  def __init__(self, filename='_autoplay.mp4', fps=30., **kwargs):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kwargs)

  def __enter__(self):
    return self
  
  def __exit__(self, *args):
    self.close()
    if self.params['filename'] == '_autoplay.mp4':
      self.show()

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in (np.float32, np.float64):
      img = np.uint8(img.clip(0, 1)*255)
    if img.ndim == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer is not None:
      self.writer.close()

  def show(self, **kwargs):
    self.close()
    fn = self.params['filename']
    display(mvp.ipython_display(fn, **kwargs))