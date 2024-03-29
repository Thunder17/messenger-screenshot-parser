import numpy as np
import cv2
from typing import List
import re
from statistics import mode
from sklearn.preprocessing import minmax_scale

from sklearn.cluster import KMeans


class Screen:
    """
    A class representing single screenshot
    Attributes:
        text_blocks (np.ndarray): ndarray of text blocks recognized by API
        sector_lines (np.ndarray): ndarray of horizontal lines dividing screen on sectors
    """
    def __init__(self, img: np.ndarray, text_blocks: List):
        self._img = img
        # self.text_blocks = text_blocks
        self.sector_lines = [0, img.shape[0]]
        # Sectors
        gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        kernel = np.tile(np.array([
            [2],
            [2],
            [-2],
            [-2]
        ]), 10)
        self.conv_horizontal1 = cv2.filter2D(src=gray, ddepth=-1, kernel=kernel)
        self.conv_horizontal2 = cv2.filter2D(src=gray, ddepth=-1, kernel=-kernel)
        means1 = np.mean(self.conv_horizontal1, axis=1)
        means2 = np.mean(self.conv_horizontal2, axis=1)
        line_inds = sorted([i for i in range(len(means1)) if means1[i] > 220] + [i for i in range(len(means2)) if means2[i] > 220])
        top_lines = [line for line in line_inds if line < img.shape[0] * 0.25]
        bot_lines = [line for line in line_inds if line > img.shape[0] * 0.75]
        if top_lines:
            self.sector_lines[0] = max(top_lines)
        if bot_lines:
            self.sector_lines[1] = min(bot_lines)
        self.text_blocks = [block for block in text_blocks if self.sector_lines[0] < block.center['y'] < self.sector_lines[1]]

    def get_img_blurred(self, ksize=33):
        return cv2.medianBlur(self._img, ksize)

    def get_img(self, draw_boxes=False, draw_sectors=False):
        if not (draw_boxes or draw_sectors):
            return self._img.copy()
        res_img = self.get_img()
        if draw_boxes:
            res_img = self._get_img_with_boxes(res_img)
        if draw_sectors:
            res_img = self._get_img_with_sectors(res_img)
        return res_img

    def get_img_gray(self):
        return cv2.cvtColor(self._img.copy(), cv2.COLOR_BGR2GRAY)

    def _get_img_with_boxes(self, img):
        img_r = img.copy()
        colors = {
            None: (200, 200, 200),
            'left': (10, 180, 10),
            'right': (10, 10, 180)
        }
        for block in self.text_blocks:
            if self.sector_lines[0] < block.center['y'] < self.sector_lines[1]:
                img_r = cv2.rectangle(
                    img_r,
                    (block.x0, block.y0),
                    (block.x1, block.y1),
                    colors[block.user],
                    thickness=5
                )
        return img_r

    def _get_img_with_sectors(self, img):
        img_r = img.copy()
        for i in self.sector_lines:
            res_img = cv2.line(img_r, (0, i), (img.shape[1], i), (255, 0, 0), 5)
        return img_r

    def recursive_blocks_join_(self):
        changed = False
        i = 0
        while i < len(self.text_blocks) - 1:
            if self.text_blocks[i].y0 < self.text_blocks[i + 1].center['y'] < self.text_blocks[i].y1\
                    or (
                        not self.text_blocks[i].ends_with_msg_time
                        and (self.text_blocks[i + 1].is_msg_time or self.text_blocks[i + 1].is_slash_and_time)
                    )\
                    or (
                        not self.text_blocks[i].ends_with_slash
                        and (self.text_blocks[i + 1].is_slash or self.text_blocks[i + 1].is_slash_and_time)
                    ):
                self.text_blocks[i].y0 = min(self.text_blocks[i].y0, self.text_blocks[i + 1].y0)
                self.text_blocks[i].y1 = max(self.text_blocks[i].y1, self.text_blocks[i + 1].y1)
                self.text_blocks[i].x0 = min(self.text_blocks[i].x0, self.text_blocks[i + 1].x0)
                self.text_blocks[i].x1 = max(self.text_blocks[i].x1, self.text_blocks[i + 1].x1)
                self.text_blocks.remove(self.text_blocks[i + 1])
                i += 2
                changed = True
                if i >= len(self.text_blocks) - 1:
                    break
            else:
                i += 1
        if changed:
            self.recursive_blocks_join_()

    def classify_users_positional(self):
        for block in self.text_blocks:
            if block.closest_left + (block.closest_right - block.closest_left) / 2 <= self._img.shape[1] * 0.49:
                block.user = 'right'
            elif block.closest_left + (block.closest_right - block.closest_left) / 2 >= self._img.shape[1] * 0.51:
                block.user = 'left'

    def classify_users_kmeans_features(self):
        gray_colors = np.array([block.get_mean_color_partial(cv2.cvtColor(self.get_img_blurred(), cv2.COLOR_BGR2GRAY)) for block in self.text_blocks])
        horizontal = np.array([block.closest_left + (block.closest_right - block.closest_left) / 2 for block in self.text_blocks])
        X = np.column_stack((
            minmax_scale(gray_colors) * 0.7,
            minmax_scale(horizontal)
        ))

        km = KMeans(n_clusters=2, random_state=17).fit(X)
        label_users = {
            0: 'left',
            1: 'right',
            2: None
        }
        if len(km.labels_) != len(self.text_blocks):
            raise ValueError('KMeans data should have same shape as text_blocks')
        for i in range(len(km.labels_)):
            self.text_blocks[i].user = label_users[km.labels_[i]]

    def filter_replies_next(self):
        left_blocks = [block for block in self.text_blocks if block.user == 'left']
        right_blocks = [block for block in self.text_blocks if block.user == 'right']

        for block_list in [left_blocks, right_blocks]:
            for i in range(len(block_list) - 1, 0, -1):
                color1 = block_list[i].get_mean_color_partial(self.get_img_blurred()).astype(int).mean()
                color2 = block_list[i - 1].get_mean_color_partial(self.get_img_blurred()).astype(int).mean()
                if ((abs(block_list[i].y0 - block_list[i - 1].y1) < block_list[i].height + block_list[i - 1].height)
                        and block_list[i].user != 'reply' and np.abs(color1 - color2) > 5):
                    block_list[i - 1].user = 'reply'
                    if i > 1:
                        color3 = block_list[i - 2].get_mean_color_partial(self.get_img_blurred()).astype(int).mean()
                        if np.abs(color1 - color3) > 7:
                            block_list[i - 2].user = 'reply'

        self.text_blocks = [block for block in self.text_blocks if block.user != 'reply']

    def check_boxes(self):
        show = self.get_img()
        gray_blur = cv2.cvtColor(self.get_img_blurred(25), cv2.COLOR_BGR2GRAY)
        kernel = np.array([2, 2, 2, -2, -2 ,-2]) * np.ones((10, 1))
        conv_vertical1 = cv2.filter2D(src=gray_blur, ddepth=-1, kernel=kernel)
        conv_vertical2 = cv2.filter2D(src=gray_blur, ddepth=-1, kernel=-kernel)
        _, thresh1 = cv2.threshold(conv_vertical1, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(conv_vertical2, 127, 255, cv2.THRESH_BINARY)
        thresh = thresh1 + thresh2
        for block in self.text_blocks:
            bot, top = int(block.center['y'] - block.height / 4), int(block.center['y'] + block.height / 4)

            left = thresh[bot:top, 0:block.x0 - 1].mean(axis=0) > 245
            right = thresh[bot:top, block.x1 + 5:self._img.shape[1]].mean(axis=0) > 245

            if left.sum() > 0:
                block.closest_left = len(left) - np.argmax(left[::-1])
            else:
                block.closest_left = 0

            if right.sum() > 0:
                block.closest_right = block.x1 + 1 + np.argmax(right)
            else:
                block.closest_right = self._img.shape[1]
            block.box_center_x = block.closest_left + (block.closest_right - block.closest_left) / 2

            show = cv2.line(show, (block.closest_left, bot), (block.closest_left, top), (100, 100, 250), 5)
            show = cv2.line(show, (block.closest_right, bot), (block.closest_right, top), (100, 100, 250), 5)
            if (block.closest_left == 0 and block.closest_right == self._img.shape[1])\
                or (block.y1 - block.y0 > self._img.shape[0] * 0.2 and len(block.text) < 50)\
                    or abs(block.box_center_x - (self._img.shape[1]/2)) < self._img.shape[1] * 0.01:
                block.user = 'trash'

        cv2.imwrite('box_detection.jpg', show)
        self.text_blocks = [block for block in self.text_blocks if block.user != 'trash']





class TextBlock:
    def __init__(self, x0: int, y0: int, x1: int, y1: int, text: str):
        self.x0 = x0
        self.y0 = y0 - 5 if y0 > 5 else y0
        self.x1 = x1
        self.y1 = y1

        self.height = self.y1 - self.y0
        self.width = self.x1 - self.x0

        self.text = text

        self.center = {'x': (x0 + x1) / 2,
                       'y': (y0 + y1) / 2}
        self.user = None

        self.is_msg_time = bool(re.fullmatch(r'\d{2}\s?:\s?\d{2}$', self.text.strip()))
        self.is_slash = bool(re.fullmatch(r'/?\s?/$', self.text.strip()))
        self.is_slash_and_time = bool(re.fullmatch(r'\d{2}\s?:\s?\d{2}\s?/?\s?/$', self.text.strip()))
        self.ends_with_msg_time = bool(re.search(r'\d{2}\s?:\s?\d{2}$', self.text.strip()))
        self.ends_with_slash = bool(re.search(r'/?\s?/$', self.text.strip()))

    def get_sector(self, img: np.ndarray):
        return img[self.y0:self.y1, self.x0:self.x1]

    def get_sector_partial(self, img: np.ndarray):
        return img[self.y0:self.y1, self.x0:(self.x0 + (self.x1 - self.x0) // 4)]

    def get_mode_color(self, img: np.ndarray):
        if len(img.shape) == 3:
            return mode_color(self.get_sector(img).reshape(-1, 3))
        elif len(img.shape) == 2:
            return mode(self.get_sector(img).ravel())
        else:
            raise ValueError("Invalid img shape, should be (n, m) or (n, m, 3)")

    def get_mean_color_partial(self, img: np.ndarray):
        if len(img.shape) == 3:
            return np.mean(self.get_sector_partial(img).reshape(-1, 3), axis=0)
        elif len(img.shape) == 2:
            return np.mean(self.get_sector_partial(img).ravel())
        else:
            raise ValueError("Invalid img shape, should be (n, m) or (n, m, 3)")


    def get_left_bottom(self, img: np.ndarray):
        return img[self.y0, self.x0]



def mode_color(a: np.ndarray):
    """Mode color (3 colors)"""
    t = [(n[0], n[1], n[2]) for n in a]
    counts = {}
    for n in t:
        counts[n] = counts.get(n, 0) + 1
    return np.array(max(counts, key=counts.get))


