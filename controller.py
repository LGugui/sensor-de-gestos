import time
from pynput.mouse import Button, Controller


class MouseController:
    def __init__(self, config):
        self.mouse = Controller()
        self.debounce = config["debounce_ms"] / 1000.0
        self.last_click = 0

    def move(self, x, y):
        self.mouse.position = (x, y)

    def click(self, button=Button.left):
        now = time.time()
        if now - self.last_click >= self.debounce:
            self.mouse.click(button)
            self.last_click = now
            return True
        return False

    def double_click(self, button=Button.left):
        self.mouse.click(button, 1)  # first click already sent via press+release
        self.last_click = time.time()
        return True

    def press(self, button=Button.left):
        self.mouse.press(button)
        self.last_click = time.time()

    def release(self, button=Button.left):
        self.mouse.release(button)

    def scroll(self, delta):
        self.mouse.scroll(0, delta)
