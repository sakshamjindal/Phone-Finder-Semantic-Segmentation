import random
from PIL import Image


def random_paste(background_image, stock_image, min_scale=0.05, max_scale=0.1):
    """Randomly scales and pastes the stock image onto the background image"""
    
    background_image = background_image.resize((256, 256))

    W, H = background_image.size
    stock_image = stock_image.resize((W, 2*W))
    w, h = stock_image.size
    # first, we will randomly downscale the iphone image
    scale = random.uniform(min_scale, max_scale)
    new_w = int(scale * w)
    new_h = int(scale * h)
    resized_stock_image = stock_image.resize((new_w, new_h))

    # second, will randomly choose the locations where to paste the new image
    start_w = random.randint(new_w, W - new_w)
    start_h = random.randint(new_h, H - new_h)

    # third, will create the blank canvas of the same size as the original image
    canvas_image = Image.new('RGBA', (W, H))

    # and paste the resized iphone onto it, preserving the mask
    canvas_image.paste(resized_stock_image, (start_w, start_h), resized_stock_image)
    
    # iphone image is of mode RGBA, while background image is of mode RGB;
    # `.paste` requires both of them to be of the same type.
    background_image = background_image.copy().convert('RGBA')
    # finally, will paste the resized iphone onto the background image
    background_image.paste(resized_stock_image, (start_w, start_h), resized_stock_image)
    return background_image, canvas_image
    
def load_labels(labels_path):

  label_data = {}

  with open(labels_path) as f: 
    lines = f.readlines()

    for line in lines:
      path, x, y = line.split()
      x = float(x); y = float(y)
      label_data[path] = [x, y]

  return label_data
