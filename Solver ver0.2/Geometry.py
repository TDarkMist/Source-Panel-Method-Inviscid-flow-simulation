from dataclasses import dataclass, field
import numpy as np


@dataclass
class Geometry:
    xc: np.ndarray
    yc: np.ndarray

    x_start: np.ndarray = field(init=False)
    y_start: np.ndarray = field(init=False)

    x_end: np.ndarray = field(init=False)
    y_end: np.ndarray = field(init=False)

    x_ctr: np.ndarray = field(init=False)
    y_ctr: np.ndarray = field(init=False)

    theta: np.ndarray = field(init=False)

    length: np.ndarray = field(init=False)

    normal: np.ndarray = field(init=False)

    panel_vec: np.ndarray = field(init=False)




    def __post_init__(self):
        #reverse the geometry to a clockwise orientation
        self.xc = self.xc
        self.yc = self.yc

        self.x_start = self.xc
        self.y_start = self.yc

        self.x_end = np.roll(self.xc, -1)
        self.y_end = np.roll(self.yc, -1)

        self.x_start = self.x_start[:-1:]
        self.y_start = self.y_start[:-1:]

        self.x_end = self.x_end[:-1:]
        self.y_end = self.y_end[:-1:]

        dx = self.x_end - self.x_start
        dy = self.y_end - self.y_start

        self.x_ctr = 0.5 * (self.x_start + self.x_end)
        self.y_ctr = 0.5 * (self.y_start + self.y_end)

        self.panel_vec = np.column_stack((dx, dy))

        self.length = np.linalg.norm(self.panel_vec, axis=1)

        self.normal = np.column_stack((-dy/self.length, dx/self.length))

        self.theta = np.arctan2(dy, dx)
