import random
import numpy as np

from PIL import Image, ImageDraw


class Circle:
    def __init__(self, coord, color, width):
        self.coord = coord
        self.color = color
        self.width = width


class Duck:
    def __init__(self, coord, dir):
        self.coord = coord
        self.dir = dir


class Pixel:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


class Drop:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color


def get_exo_util(exo_name, config):
    if exo_name == CircleExogenous.NAME:
        return CircleExogenous(config)

    elif exo_name == PixelNoise.NAME:
        return PixelNoise(config)

    elif exo_name == DropNoise.NAME:
        return DropNoise(config)

    else:
        raise AssertionError("Exogenous noise %s not found." % exo_name)


class ExoUtil:
    def reset(self):
        """
        Reset exogenous variables
        :return: None
        """
        raise NotImplementedError()

    def update(self):
        """
        Update exogenous variable
        :return: None
        """
        raise NotImplementedError()

    def generate_image(self, img):
        """Update image img using exogenous noise
        :param img: Input image without exogenous noise
        :return: Add exogenous noise to the input image and return it along with exogenous noise
        """
        raise NotImplementedError()

    def get_reward(self):
        return 0.0


class CircleExogenous(ExoUtil):
    NAME = "circle"

    def __init__(self, config):
        self.ego_centric = config["ego_centric"] > 0

        if self.ego_centric:
            self.height, self.width = (
                config["agent_view_size"] * config["tile_size"],
                config["agent_view_size"] * config["tile_size"],
            )
        else:
            self.height, self.width = (
                config["height"] * config["tile_size"],
                config["width"] * config["tile_size"],
            )

        self.colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(8)
        ]
        self.circle_width = config["circle_width"]
        self.circle_motion = config["circle_motion"]
        self.num_circles = config["num_exo_var"]
        self.circles = None

    def reset(self):
        self.circles = []
        for _ in range(0, self.num_circles):
            # Generate random four points
            coord = [
                random.randint(0, self.width // 2),
                random.randint(0, self.height // 2),
                random.randint(self.width // 2, self.width),
                random.randint(self.height // 2, self.height),
            ]
            color = random.choice(self.colors)
            self.circles.append(Circle(coord, color, self.circle_width))

    def update(self):
        self.circles = [
            self._perturb_circle(circle, self.height, self.width)
            for circle in self.circles
        ]

    def _perturb_circle(self, circle, height, width):
        # Each of the four coordinate is moved independently by 10% of the corresponding dimension
        r = [random.choice([-1, 1]) for _ in range(4)]
        coord = (
            circle.coord[0] + r[0] * int(self.circle_motion * width),
            circle.coord[1] + r[1] * int(self.circle_motion * height),
            circle.coord[2] + r[2] * int(self.circle_motion * width),
            circle.coord[3] + r[3] * int(self.circle_motion * height),
        )

        return Circle(coord=coord, color=circle.color, width=circle.width)

    def generate_image(self, img):
        width, height = img.shape[0], img.shape[1]
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)

        for circle in self.circles:
            draw.ellipse(circle.coord, outline=circle.color, width=circle.width)
        exo_im = np.array(image).astype(np.uint8)

        img_shape = img.shape
        exo_im = exo_im.reshape((-1, 3))
        img = img.reshape((-1, 3))
        obs_max = img.max(1)
        bg_pixel_ix = np.argwhere(
            obs_max < 100
        )  # flattened (x, y) position where pixels are black in color
        values = np.squeeze(exo_im[bg_pixel_ix])
        np.put_along_axis(img, bg_pixel_ix, values, axis=0)
        img = img.reshape(img_shape)

        return img, None


class PixelNoise(ExoUtil):
    NAME = "pixel"

    def __init__(self, config):
        self.ego_centric = config["ego_centric"] > 0
        self.size = config["pixel_size"]
        self.horizon = config["horizon"]

        if self.ego_centric:
            self.height, self.width = (
                config["agent_view_size"] * config["tile_size"],
                config["agent_view_size"] * config["tile_size"],
            )
        else:
            self.height, self.width = (
                config["height"] * config["tile_size"],
                config["width"] * config["tile_size"],
            )

        self.num_pixels = config["num_exo_var"]
        self.pixels = None

    def reset(self):
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(self.num_pixels)
        ]
        self.pixels = []

        for i in range(self.num_pixels):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            pixel = Pixel(x, y, self._to_rgb(colors[i]))
            self.pixels.append(pixel)

    @staticmethod
    def _to_rgb(hex):
        return tuple(int(hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    def update(self):
        self.pixels = [
            self._perturb_pixels(pixel, self.height, self.width)
            for pixel in self.pixels
        ]

    @staticmethod
    def _perturb_pixels(pixel, height, width):
        rnd = [random.choice([-1, 1]) for _ in range(5)]

        new_x = (pixel.x + rnd[0] * int(0.1 * width)) % width
        new_y = (pixel.y + rnd[1] * int(0.1 * height)) % height
        (
            r,
            g,
            b,
        ) = pixel.color

        new_color = (
            (r + rnd[2] * 5) % 255,
            (g + rnd[3] * 5) % 255,
            (b + rnd[4] * 5) % 255,
        )

        return Pixel(x=new_x, y=new_y, color=new_color)

    def generate_image(self, img):
        exo_noise = np.zeros(img.shape).astype(np.uint8)

        for pixel in self.pixels:
            for i in range(pixel.x - self.size // 2, pixel.x + self.size // 2 + 1):
                for j in range(pixel.y - self.size // 2, pixel.y + self.size // 2 + 1):
                    if (
                        0 <= i < img.shape[0]
                        and 0 <= j < img.shape[1]
                        and np.max(img[i, j, :]) < 100
                    ):  # Pixel is black in color denoting background
                        img[i, j, :] = pixel.color
                        exo_noise[i, j, :] = pixel.color

        return img, exo_noise

    def get_reward(self):
        if random.random() > 1 / float(self.horizon):
            return 0.0
        else:
            score = 0.0
            for pixel in self.pixels:
                score_ = (
                    pixel.x / float(self.width)
                    + pixel.y / float(self.height)
                    + pixel.color[0] / 255.0
                    + pixel.color[1] / 255.0
                    + pixel.color[2] / 255.0
                )
                score += score_ / 5.0

            score /= float(len(self.pixels))

            if score > 0.5:
                return max(min(random.random() + 0.5, 1), 0)
            else:
                return max(min(random.random() - 0.5, 1), 0)


class DropNoise(ExoUtil):
    NAME = "drop"

    def __init__(self, config):
        self.ego_centric = config["ego_centric"] > 0
        self.size = config["pixel_size"]
        self.horizon = config["horizon"]

        if self.ego_centric:
            self.height, self.width = (
                config["agent_view_size"] * config["tile_size"],
                config["agent_view_size"] * config["tile_size"],
            )
        else:
            self.height, self.width = (
                config["height"] * config["tile_size"],
                config["width"] * config["tile_size"],
            )

        self.num_drops = config["num_exo_var"]
        self.drops = None

    def reset(self):
        colors = [
            "#" + "".join([random.choice("0123456789ABCDEF") for _ in range(6)])
            for _ in range(self.num_drops)
        ]
        self.drops = []

        for i in range(self.num_drops):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            drop = Drop(x, y, self._to_rgb(colors[i]))
            self.drops.append(drop)

    @staticmethod
    def _to_rgb(hex):
        return tuple(int(hex.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4))

    def update(self):
        self.drops = [
            self._perturb_drops(drop, self.height, self.width) for drop in self.drops
        ]

    @staticmethod
    def _perturb_drops(drop, height, width):
        rnd = [random.choice([-1, 1]) for _ in range(5)]

        new_x = (drop.x + rnd[0] * int(0.1 * width)) % width
        new_y = (drop.y + rnd[1] * int(0.1 * height)) % height
        (
            r,
            g,
            b,
        ) = drop.color

        new_color = (
            (r + rnd[2] * 5) % 255,
            (g + rnd[3] * 5) % 255,
            (b + rnd[4] * 5) % 255,
        )

        return Drop(x=new_x, y=new_y, color=new_color)

    def generate_image(self, img):
        exo_noise = np.zeros(img.shape).astype(np.uint8)

        for drop in self.drops:
            # Drops are different from pixels as follows:
            # If we have N size, then the size increases from 1 to N for the top N/2 and then shrinks to 1
            for i in range(drop.y - self.size // 2, drop.y + self.size // 2 + 1):
                # Find the width of the drop
                if i <= drop.y:
                    # 1 at i = drop.y - self.size // 2 and self.size + 1 at i = drop.y
                    cols = 2 * (i - drop.y + self.size // 2) + 1
                else:
                    # self.size + 1 at i = drop.y and 1 at i = drop.y + self.size // 2
                    cols = 2 * (drop.y + self.size // 2 - i) + 1

                for j in range(drop.x - cols // 2, drop.x + cols // 2 + 1):
                    if (
                        0 <= i < img.shape[0]
                        and 0 <= j < img.shape[1]
                        and np.max(img[i, j, :]) < 100
                    ):  # Pixel is black in color denoting background
                        img[i, j, :] = drop.color
                        exo_noise[i, j, :] = drop.color

        return img, exo_noise

    def get_reward(self):
        if random.random() > 1 / float(self.horizon):
            return 0.0
        else:
            score = 0.0
            for drop in self.drops:
                score_ = (
                    drop.x / float(self.width)
                    + drop.y / float(self.height)
                    + drop.color[0] / 255.0
                    + drop.color[1] / 255.0
                    + drop.color[2] / 255.0
                )
                score += score_ / 5.0

            score /= float(len(self.drops))

            if score > 0.5:
                return max(min(random.random() + 0.5, 1), 0)
            else:
                return max(min(random.random() - 0.5, 1), 0)
