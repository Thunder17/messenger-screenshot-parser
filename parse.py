import numpy as np
import cv2
import json

from objects import Screen, TextBlock, mode_color


def parse(img_name, jsons_path, images_path, save_path):

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
    screen.check_boxes()
    screen.classify_users_positional()


    # for block in screen.text_blocks:
    #     if block.user == 'right':
    #         print(f'{block.text} -- {block.get_mean_color_partial(screen.get_img_gray())} ')

    screen.filter_replies_next()
    cv2.imshow('1', screen.get_img(True, True))
    cv2.waitKey(0)
    cv2.imwrite(save_path + img_name, screen.get_img(True, True))
