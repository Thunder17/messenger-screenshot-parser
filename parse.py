import numpy as np
import cv2
import json

from objects import Screen, TextBlock, mode_color

img_name = 'test_img.jpg'
images_path = 'test_images/'
jsons_path = 'test_jsons/'

f = open(jsons_path + img_name + '.json', 'r')
img = cv2.imread(images_path + img_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
data = json.load(f)

textblocks = []
for block in data['textAnnotation']['blocks']:
    veritces = block['boundingBox']['vertices']
    x0 = int(veritces[0]['x'])
    y0 = int(veritces[0]['y'])
    x1 = int(veritces[2]['x'])
    y1 = int(veritces[2]['y'])
    text = ' '.join([line['text'] for line in block['lines']])
    textblocks.append(
        TextBlock(x0=x0, y0=y0, x1=x1, y1=y1, text=text)
    )


screen = Screen(
    img,
    textblocks
)
screen.recursive_blocks_join_()
screen.filter_trash()
screen.classify_users_kmeans()


for block in screen.text_blocks:
    if block.user == 'right':
        print(f'{block.text} -- {block.get_mean_color(img)} ')

screen.filter_replies_next()

cv2.imshow('km', screen.get_img(True, True))
cv2.waitKey(0)
