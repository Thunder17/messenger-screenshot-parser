from api import get_response
from parse import parse
from sys import argv

import os

img_name = 'test_img_4.jpg'
images_path = 'test_images/'
jsons_path = 'test_jsons/'
marked_jsons_path = 'test_jsons_marked/'
save_path = 'test_images_parsed/'

# for img_name in os.listdir('test_images'):
#     #get_response(images_path + img_name)
#     parse(img_name, jsons_path, images_path, save_path)
#     print(img_name)
print(parse(img_name, jsons_path, images_path, save_path))