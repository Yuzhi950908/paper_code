import math
import os
import sys
import numpy as np
import pyglet
from pyglet.gl import gl


# 这个文件 我刚把 整个环境的网格画出来


_GRID_COLOR=(0,0,0)
_BACKGROUND_COLOR=(255,255,255)
class Viewer(object):
    def __init__(self,world_size):
        self.rows, self.cols = world_size

        self.grid_size= 30
        self.icon_size= 20
        self.width= 1 + self.cols*(self.grid_size+1)
        self.height= 1 + self.rows* (self.grid_size+1)
        self.window = pyglet.window.Window(
            width=self.width, height=self.height, display=None
        )
        self.window.on_close = self.window_closed_by_user
        self.isopen=True
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA,gl.GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.isopen=False
        pyglet.app.exit()
    #def render(self, env, return_rgb_array=False):
        ###

    def _draw_grid(self):
        batch=pyglet.graphics.Batch()
        for r in range(self.rows+1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    'v2f',
                    (
                        1,
                        (self.grid_size+1)*r+1,
                        (self.grid_size+1)*self.cols+1,
                        (self.grid_size+1)*r+1
                    ),
                ),
                ("c3B",(*_GRID_COLOR,*_GRID_COLOR)),
            )
        for c in range(self.cols+1):
            batch.add(
                2,
                gl.GL_LINES,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size+1)*c+1,
                        1,
                        (self.grid_size+1)*c+1,
                        (self.grid_size+1)*self.rows+1,

                    ),
                ),
                ("c3B",(*_GRID_COLOR,*_GRID_COLOR)),
            )
        batch.draw()


