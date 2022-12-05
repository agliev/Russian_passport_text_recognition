from random import choice
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import cv2
import os
from source.functions import gender_recogniser, adaptive_gaussian_binarization, gen_rand_info



font_list = os.listdir('fonts/')
font = ImageFont.truetype(f'fonts/{choice(font_list)}', 19)
rgb = (56, 54, 54)

def pass_gen(pass_base_path:str='pass_base.jpg',
             img_save:bool=False,
             pass_folder_path:str='genered_pass_folder/'):
  
  path, format = pass_base_path.split('.')
  res_file_path = f'{path}_resized.{format}'

  if os.path.isfile(res_file_path):
    img = cv2.imread(res_file_path)
    img = Image.fromarray(img)
    
  else:
    img = cv2.imread(pass_base_path)
    img = cv2.resize(img, (600,903))

    img = Image.fromarray(img)
    img.save(f'{path}_resized.{format}')
  
  draw = ImageDraw.Draw(img)

  name, patronymic, surname, birth_date, city, adm_unit = gen_rand_info()
  birth_date = birth_date.replace('.',' . ')
  
  draw.text((340, 510), surname, rgb, font=font)
  draw.text((350, 570), name, rgb, font=font)
  draw.text((337, 603), patronymic, rgb, font=font)
  draw.text((235, 635), gender_recogniser(patronymic), rgb, font=font)
  draw.text((350, 633), birth_date, rgb, font=font)
  draw.text((330, 663), city, rgb, font=font)
  draw.text((330, 693), adm_unit, rgb, font=font)

  binary = adaptive_gaussian_binarization(np.array(img))


  if img_save == True:

    binary.save(f"{pass_folder_path}{surname} {name} {patronymic}.{format}")
    # binary.save(f"{pass_folder_path}{surname} {name} {patronymic} {birth_date} {city} {adm_unit}.{format}")

  return binary

