from api import get_response
from parse import parse

import os

img_name = 'test_img.jpg'
images_path = 'test_images/'
jsons_path = 'test_jsons/'
save_path = 'test_images_parsed/'

# for img_name in os.listdir('test_images'):
#     #get_response(images_path + img_name)
#     parse(img_name, jsons_path, images_path, save_path)
#     print(img_name)
parse(img_name, jsons_path, images_path, save_path)