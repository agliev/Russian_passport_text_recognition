import numpy as np
import cv2
from PIL import Image
import randominfo as ri
from russian_names import RussianNames
from faker import Faker
from faker.providers.address.ru_RU import Provider

genders = ['МУЖ', "ЖЕН"]

def gender_recogniser(patronimic:str,
                      genders:list=genders)->str:

  if patronimic[-3:] == 'вич':
    return genders[0]
  
  elif patronimic[-3:] == 'вна':
    return genders[1]
  
  else:
    return genders[1]


def adaptive_gaussian_binarization(image:np.array,
                                   block_size:int=21) -> np.array: 

  img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  binary = cv2.adaptiveThreshold(img_gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                        cv2.THRESH_BINARY,block_size,30)
  binary = Image.fromarray(binary)

  return binary


fake = Faker()
fake.add_provider(Provider)

def gen_rand_info() -> tuple:

    name, patronymic, surname = RussianNames().get_person().split()
    birth_date = ri.get_birthdate(_format = '%d.%m.%Y')
    city = fake.city()
    adm_unit = fake.administrative_unit()

    return name, patronymic, surname, birth_date, city, adm_unit
