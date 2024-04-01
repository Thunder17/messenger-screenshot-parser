from api import get_response
from parse import parse
from sys import argv

import os

img_name = None
save_path = 'test_images_parsed/'

resp = get_response(img_name, True)
parse(resp, img_name, save_path)