import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont


class AppObservation:
    """
    AppObservation is a "clickable screenshot"
    """

    def __init__(self, size=512):
        self.ui_elements = []
        self.screenshot = None
        self.size = size

    def get_screenshot(self):
        """
        Return screenshot as PIL Image object
        """
        if self.screenshot is None:
            self._render_screenshot()
        return self.screenshot

    def get_action_at_click(self, x, y):
        """
        Return action at click location (x, y)
        Returns None if no action is associated with the clicked UI element, or nothing is clicked
        """
        for ui_element in self.ui_elements:
            a = ui_element.get_action_at_click(x, y)
            if a is not None:
                return a
        return None

    def get_all_clickable_actions(self):
        """
        Return a list of all actions associated with UI elements
        There may be additional actions possible but not associated with UI elements
        """
        actions = []
        for ui_element in self.ui_elements:
            a = ui_element.get_action()
            if a is not None:
                actions.append(a)
        return actions

    def add_ui_element(self, ui_element):
        """
        Add a UI element to the observation
        """
        self.ui_elements.append(ui_element)
        # invalidate screenshot
        self.screenshot = None

    def _render_screenshot(self):
        self.screenshot = Image.new("RGB", (self.size, self.size), color=(255, 255, 255))
        draw = ImageDraw.Draw(self.screenshot)
        for ui_element in self.ui_elements:
            ui_element.render(draw)


class UiElement:
    """
    Base class for UI elements
    (x0, y0) is the top left corner of bounding box
    (x1, y1) is the bottom right corner of bounding box
    """

    def __init__(self, x_center, y_center, width, height, action):
        self.x0 = x_center - width // 2
        self.y0 = y_center - height // 2
        self.x1 = self.x0 + width - 1
        self.y1 = self.y0 + height - 1
        self.action = action

    def get_bounding_box(self):
        return (self.x0, self.y0, self.x1, self.y1)

    def contains_point(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def get_action(self):
        return self.action

    def get_action_at_click(self, x, y):
        if self.contains_point(x, y):
            return self.action
        return None

    def render(self, draw):
        raise NotImplementedError


class ToggleSwitch(UiElement):
    """
    Draw a toggle switch with the given display_state
    True = on
    False = off
    """

    def __init__(self, x_center, y_center, height, action=None, display_state=False):
        super().__init__(x_center, y_center, height * 2, height, action)
        self.display_state = display_state
        self.height = height

    def render(self, draw):
        corner_radius = self.height // 2
        if self.display_state:
            # switch is on
            draw.rounded_rectangle(
                [self.x0, self.y0, self.x1, self.y1], fill=(0, 0, 0), outline=(0, 0, 0), radius=corner_radius
            )
            draw.ellipse([self.x0 + self.height, self.y0, self.x1, self.y1], fill=(255, 255, 255), outline=(0, 0, 0))
        else:
            # switch is off
            draw.rounded_rectangle(
                [self.x0, self.y0, self.x1, self.y1], fill=(255, 255, 255), outline=(0, 0, 0), radius=corner_radius
            )
            draw.ellipse([self.x0, self.y0, self.x0 + self.height, self.y1], fill=(0, 0, 0), outline=(0, 0, 0))


class Text(UiElement):
    """
    Draw text
    """

    def __init__(self, x_center, y_center, text, font_size=12, action=None, color=(0, 0, 0)):
        super().__init__(x_center, y_center, 0, 0, action)
        self.x_center = x_center
        self.y_center = y_center
        self.text = text
        font_path = fm.findfont("monospace")
        self.font = ImageFont.truetype(font_path, font_size)
        self.color = color

        bbox = self.font.getbbox(self.text, anchor="mm")
        self.x0 = x_center + bbox[0]
        self.y0 = y_center + bbox[1]
        self.x1 = x_center + bbox[2]
        self.y1 = y_center + bbox[3]

    def render(self, draw):
        # draw.rectangle([self.x0, self.y0, self.x1, self.y1], outline=(0, 0, 0))
        draw.text((self.x_center, self.y_center), self.text, font=self.font, anchor="mm", fill=self.color)


class Button(UiElement):
    """
    Draw a button with optional text
    """

    def __init__(self, x_center, y_center, width=0, height=0, action=None, text="", font_size=12, enabled=True):
        super().__init__(x_center, y_center, width, height, action)
        self.x_center = x_center
        self.y_center = y_center
        self.enabled = enabled
        self.action = action if enabled else None
        self.color = (0, 0, 0) if enabled else (200, 200, 200)

        if text:
            self.text = Text(x_center, y_center, text, font_size, action=None, color=self.color)

            # if size is 0, use text size
            if width == 0:
                width = int(1.5 * (self.text.x1 - self.text.x0))
            if height == 0:
                height = int(1.5 * (self.text.y1 - self.text.y0))

            # recalculate bounding box
            self.x0 = x_center - width // 2
            self.y0 = y_center - height // 2
            self.x1 = self.x0 + width - 1
            self.y1 = self.y0 + height - 1

        else:
            self.text = None

    def render(self, draw):
        corner_radius = min(self.x1 - self.x0, self.y1 - self.y0) // 4
        draw.rounded_rectangle(
            [self.x0, self.y0, self.x1, self.y1], fill=(255, 255, 255), outline=self.color, radius=corner_radius
        )

        if self.text is not None:
            self.text.render(draw)
