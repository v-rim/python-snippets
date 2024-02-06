import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# https://matplotlib.org/stable/users/explain/animations/blitting.html
class PointsInSpace:
    def __init__(
        self,
        title=None,
        xlabel=None,
        ylabel=None,
        x_lim=[-1.5, 1.5],
        y_lim=None,
        m="o",
        hide_axis=False,
        tight=False,
        enable_grid=False,
    ):
        matplotlib.rcParams["toolbar"] = "None"
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()

        if y_lim is None:
            y_lim = x_lim

        self.ax.set_xlim(x_lim)
        self.ax.set_ylim(y_lim)
        self.ax.set_aspect("equal", adjustable="box")

        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if hide_axis:
            self.ax.set_axis_off()
        # From https://stackoverflow.com/a/47893499
        # This also lowers performance for some reason
        if tight:
            self.fig.tight_layout(pad=0)
            w, h = self.fig.get_size_inches()
            x1, x2 = self.ax.get_xlim()
            y1, y2 = self.ax.get_ylim()
            self.fig.set_size_inches(
                1.1 * w, self.ax.get_aspect() * (y2 - y1) / (x2 - x1) * w
            )
        if enable_grid:
            self.ax.grid()

        # Not sure why assignment needs to be like this but it does
        (self.points,) = self.ax.plot(0, 0, m, animated=True)
        plt.show(block=False)
        plt.pause(0.1)
        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.points)
        self.fig.canvas.blit(self.fig.bbox)

    def draw_points(self, x, y, delay=0):
        self.fig.canvas.restore_region(self.bg)
        self.points.set_data(x, y)

        self.ax.draw_artist(self.points)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

        if delay > 0:
            plt.pause(delay)


if __name__ == "__main__":
    pp = PointsInSpace("Dots circling")
    frame_count = 500
    num_dots = 15
    speed = 0.01

    tic = time.time()
    for i in range(frame_count):
        t = (2 * np.pi / num_dots) * np.arange(num_dots)
        t += i * speed

        x = np.cos(t) * np.cos(4 * t)
        y = np.sin(t) * np.cos(4 * t)

        pp.draw_points(x, y)

    print(f"Average FPS: {frame_count / (time.time() - tic)}")
