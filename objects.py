import numpy as np
import cv2
from typing import List
import re
from statistics import mode
from sklearn.preprocessing import minmax_scale
from dbscan import DBSCAN

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
        self.scanned_img = None
        self.scanned_labels = None
        gray = cv2.cvtColor(self._img, cv2.COLOR_BGR2GRAY)
        horizontal = np.array([
            [2],
            [2],
            [0],
            [-2],
            [-2]
        ]) * np.ones((1, 6))
        vertical = np.array([
            2, 2, 0 -2, -2
        ]) * np.ones((6, 1))
        kernel = np.array([
            [-1, -1,-1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

        _, conv_h_1 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=horizontal), 127, 255, cv2.THRESH_BINARY)
        _, conv_h_2 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=-horizontal), 127, 255, cv2.THRESH_BINARY)
        _, conv_v_1 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=vertical), 127, 255, cv2.THRESH_BINARY)
        _, conv_v_2 = cv2.threshold(cv2.filter2D(src=gray, ddepth=-1, kernel=-vertical), 127, 255, cv2.THRESH_BINARY)

        self.edges = conv_h_1 + conv_h_2 + conv_v_1 + conv_v_2
        self.edges_h = conv_h_1 + conv_h_2
        self.sector_lines = [0, img.shape[0]]

        means1 = np.mean(self.edges_h, axis=1)
        line_inds = sorted([i for i in range(len(means1)) if means1[i] > 220])
        top_lines = [line for line in line_inds if line < img.shape[0] * 0.25]
        bot_lines = [line for line in line_inds if line > img.shape[0] * 0.75]
        if top_lines:
            self.sector_lines[0] = max(top_lines)
        if bot_lines:
            self.sector_lines[1] = min(bot_lines)
        self.text_blocks = [block for block in text_blocks if self.sector_lines[0] < block.center['y'] < self.sector_lines[1]]
        if re.match(r'^сообщение.*', self.text_blocks[-1].text.strip().lower()) or re.match(r'^сообщение.*', self.text_blocks[-1].text.strip().lower()):
            self.text_blocks = self.text_blocks[:-1]
        

    def get_img_blurred(self, ksize=33, gray=False):
        """
        Returns a blurred version of the screenshot image.

        Args:
            ksize (int, optional): The kernel size for the median blur filter. Defaults to 33.
            gray (bool, optional): If True, the image is converted to grayscale before applying the blur filter. Defaults to False.

        Returns:
            np.ndarray: The blurred image as a NumPy array.
        """
        if gray:
            return cv2.cvtColor(cv2.medianBlur(self._img, ksize), cv2.COLOR_RGB2GRAY)
        return cv2.medianBlur(self._img, ksize)

    def get_img(self, draw_boxes=False, draw_sectors=False):
        """
        Returns the original or a modified version of the screenshot image.

        Args:
            draw_boxes (bool, optional): If True, bounding boxes around detected text blocks are drawn on the image. Defaults to False.
            draw_sectors (bool, optional): If True, horizontal lines dividing the screen into sectors are drawn on the image. Defaults to False.

        Returns:
            np.ndarray: The original or modified image as a NumPy array.
        """
        if not (draw_boxes or draw_sectors):
            return self._img.copy()
        res_img = self.get_img()
        if draw_boxes:
            res_img = self._get_img_with_boxes(res_img)
        if draw_sectors:
            res_img = self._get_img_with_sectors(res_img)
        return res_img

    def get_img_gray(self):
        """
        Returns a grayscale version of the screenshot image.

        Returns:
            np.ndarray: The grayscale image as a NumPy array.
        """
        return cv2.cvtColor(self._img.copy(), cv2.COLOR_BGR2GRAY)

    def _get_img_with_boxes(self, img):
        """
        Draws bounding boxes around detected text blocks on the image.

        Args:
            img (np.ndarray): The image to draw bounding boxes on.

        Returns:
            np.ndarray: The image with bounding boxes drawn on it.
        """
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
                    (block.edge_x0, block.edge_y0),
                    (block.edge_x1 + 3, block.edge_y1),
                    colors[block.user],
                    thickness=5
                )
        return img_r

    def _get_img_with_sectors(self, img):
        """
        Draws horizontal lines dividing the screen into sectors on the image.

        Args:
            img (np.ndarray): The image to draw sector lines on.

        Returns:
            np.ndarray: The image with sector lines drawn on it.
        """
        img_r = img.copy()
        for i in self.sector_lines:
            res_img = cv2.line(img_r, (0, i), (img.shape[1], i), (255, 0, 0), 5)
        return img_r

    def recursive_blocks_join_(self):
        """
        Merges consecutive text blocks that are likely part of the same message.

        This is done by recursively checking the following conditions:

        - If the Y coordinates of two consecutive blocks overlap.
        - If the first block does not end with a message timestamp or slash, and the second block is either a timestamp or a slash.
        - If the first block does not end with a slash, and the second block is either a slash or a timestamp with a slash.
        (OCR treat the sign of sent/read/unread message as a slash or 2 slahes)

        If any of these conditions are met, the two blocks are merged. This process is repeated until no further merging can be done.
        """
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

    def scan(self):
        """
        Performs text segmentation on the screenshot image using DBSCAN clustering.

        DBSCAN is then applied to this feature array with a fixed epsilon value and minimum number of samples required to form a cluster.
        The resulting labels are used to identify different text regions in the image.

        The scanned image and labels are stored as `scanned_img` and `scanned_labels` attributes of the `Screen` object respectively.
        """
        img = self.get_img_blurred(ksize=21)
        shape = img.shape
        X = np.zeros((shape[0] * shape[1], 5))
        for i in range(shape[0]):
            for j in range(shape[1]):
                X[shape[1] * i + j] = (i  / shape[0] , j / shape[1], *(img[i, j] / 255))

        scan = DBSCAN(X, eps=0.05, min_samples=1000)
        self.scanned_img = (scan[0] / max(scan[0]) * 255).astype(np.uint8).reshape(shape[:2])
        self.scanned_labels = scan[0].reshape(shape[:2])

    def detect_edges_scanned(self):
        """
        Detects edges of text blocks based on the labels obtained from DBSCAN text segmentation.

        For each text block, the following steps are performed:

        1. The label of the pixel at the top-left corner of the block is used as a reference.
        2. The horizontal and vertical profiles of the label image are computed within the bounding box of the text block.
        3. The edges of the text block are estimated by finding the points where the horizontal and vertical profiles deviate significantly from the reference label.

        """
        for block in self.text_blocks:
            try:
                block.scanned_label = self.scanned_labels[block.y0 - 2, block.center['x']]
                horizontal = (self.scanned_labels[block.y0:block.y1, :] == block.scanned_label).astype(int).mean(axis=0)
                left, right = horizontal[0:block.x0], horizontal[block.x1 + 1:self._img.shape[1]]
                a, b = np.where(left < 0.7)[0][-1], np.where(right < 0.7)[0][0]
                block.edge_x0, block.edge_x1 = a, block.x1 + b
                try:
                    vertical = (self.scanned_labels[:, block.edge_x0:block.edge_x1] == block.scanned_label).astype(int).mean(axis=1)
                    top, bot = vertical[:block.y0], vertical[block.y1:]
                    c, d = np.where(top < 0.8)[0][-1], np.where(bot < 0.8)[0][0]
                    block.edge_y0, block.edge_y1 = c, block.y1 + d
                except:
                    print(block.text)
                    block.edge_y0, block.edge_y1 = block.y0, block.y1
            except:
                block.edge_x0, block.edge_x1 = block.x0, block.x1
                block.edge_y0, block.edge_y1 = block.y0, block.y1
        for i in range(len(self.text_blocks) - 1):
            if self.text_blocks[i].edge_y1 > self.text_blocks[i + 1].edge_y0:
                div_line = (self.text_blocks[i].y1 + self.text_blocks[i + 1].y0) // 2
                self.text_blocks[i].edge_y1 = div_line
                self.text_blocks[i + 1].edge_y0 = div_line


    def classify_users_positional(self):
        for block in self.text_blocks:
            if block.edge_x0 < self._img.shape[1] * 0.25:
                block.user = 'left'
            elif block.edge_x1 > self._img.shape[1] * 0.75:
                block.user = 'right'
            else:
                block.user = None

    def filter_replies_next(self):
        left_blocks = [block for block in self.text_blocks if block.user == 'left']
        right_blocks = [block for block in self.text_blocks if block.user == 'right']

        for block_list in [left_blocks, right_blocks]:
            for i in range(len(block_list) - 1, 0, -1):
                color1 = block_list[i].get_left_bottom(self.get_img_blurred(25, gray=True)).astype(int)
                color2 = block_list[i - 1].get_left_bottom(self.get_img_blurred(25, gray=True)).astype(int)
                if ((abs(block_list[i].y0 - block_list[i - 1].y1) < block_list[i].height + block_list[i - 1].height)
                        and block_list[i].user != 'reply' and np.abs(color1 - color2) > 10):
                    block_list[i - 1].user = 'reply'
                    if i > 1:
                        color3 = block_list[i - 2].get_left_bottom(self.get_img_blurred()).astype(int).mean()
                        if np.abs(color2 - color3) < 4:
                            block_list[i - 2].user = 'reply'

        self.text_blocks = [block for block in self.text_blocks if block.user != 'reply']

class TextBlock:
    def __init__(self, x0: int, y0: int, x1: int, y1: int, text: str):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.edge_x0 = None
        self.edge_x1 = None
        self.edge_y0 = None
        self.edge_y1 = None

        self.height = self.y1 - self.y0
        self.width = self.x1 - self.x0

        self.text = text

        self.center = {'x': (x0 + x1) // 2,
                       'y': (y0 + y1) // 2}
        self.user = None

        self.is_msg_time = bool(re.fullmatch(r'\d{2}\s?:\s?\d{2}$', self.text.strip()))
        self.is_slash = bool(re.fullmatch(r'/?\s?/$', self.text.strip()))
        self.is_slash_and_time = bool(re.fullmatch(r'\d{2}\s?:\s?\d{2}\s?/?\s?/$', self.text.strip()))
        self.ends_with_msg_time = bool(re.search(r'\d{2}\s?:\s?\d{2}$', self.text.strip()))
        self.ends_with_slash = bool(re.search(r'/?\s?/$', self.text.strip()))

        self.scan_label = None

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


