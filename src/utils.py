import cv2

from objects import Screen, TextBlock, mode_color


def parse(data, img_name, save_path):

    # f = open(jsons_path + img_name + '.json', 'r')
    img = cv2.imread(img_name)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # data = json.load(f)

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
    screen.scan()
    screen.detect_edges_scanned()
    screen.classify_users_positional()
    screen.filter_replies_next()
    cv2.imwrite(save_path + img_name, screen.get_img(True, True))
    cv2.imshow('1.jpg', screen.get_img(True, True))
    cv2.imshow('efefef', screen.scanned_img)

    cv2.waitKey(0)
    return {str(i): {'user': block.user, 'text': block.text} for i, block in enumerate(screen.text_blocks)}