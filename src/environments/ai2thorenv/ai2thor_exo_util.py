import random
import numpy as np

from PIL import Image, ImageDraw


class AI2ThorExoUtil:

    def __init__(self, config):

        self.width, self.height, self.channel = config["obs_dim"]

        self.circle_width = int(0.05 * self.width)

        # There is a circle, square, rectangle, triangle
        self.circle_coord = None
        self.circle_color = None
        self.circle_rad = int(0.15 * self.width)

        self.triangle_x = None
        self.triangle_y = None
        self.triangle_color = None

    def reset(self):

        colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for _ in range(6)]) for _ in range(2)]

        self.circle_coord = int(random.random() * self.width), int(random.random() * self.height)
        self.circle_color = self._to_rgb(colors[1])

        self.triangle_x = int(random.random() * self.width)
        self.triangle_y = int(random.random() * self.height)
        self.triangle_color = self._to_rgb(colors[0])

    @staticmethod
    def _to_rgb(hex):
        return tuple(int(hex.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    def update(self):

        rnd = [random.choice([-1, 1]) for _ in range(4)]

        self.circle_coord = (self.circle_coord[0] + rnd[0] * int(0.1 * self.width)) % self.width, \
                            (self.circle_coord[1] + rnd[1] * int(0.1 * self.height)) % self.height

        self.triangle_x = (self.triangle_x + rnd[2] * int(0.1 * self.width)) % self.width
        self.triangle_y = (self.triangle_y + rnd[3] * int(0.1 * self.height)) % self.height

    def generate(self, img):

        width, height = img.shape[0], img.shape[1]

        image = Image.new('RGB', (2 * width, height))
        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (self.circle_coord[0] - self.circle_rad, self.circle_coord[1] - self.circle_rad,
             self.circle_coord[0] + self.circle_rad, self.circle_coord[1] + self.circle_rad),
            outline=self.circle_color, width=self.circle_width)

        draw.polygon([(self.triangle_x, self.triangle_y),
                      (int(0.15 * self.width) + self.triangle_x, self.triangle_y),
                      (int(0.075 * self.width) + self.triangle_x, int(0.075 * self.height) + self.triangle_y)],
                     fill=self.triangle_color)

        gen_img = np.array(image).astype(np.uint8)

        gen_img[:, width:, :] = img

        gen_img = gen_img / 255.0

        return gen_img
