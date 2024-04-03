from api import get_response
from utils import parse
from sys import argv
import json

img_name = None
save_path = '../data/test_images_parsed/'

if len(argv) == 2:
    img_name = argv[-1]
elif len(argv) == 3:
    img_name = argv[-2]
    save_path = argv[-1]
else:
    raise Exception('Invalid arguments')


resp = get_response(img_name, True)
with open(save_path + '/' + img_name.split('/')[-1] + 'json', 'w') as f:
    f.write(json.dumps(parse(resp, img_name, save_path)))