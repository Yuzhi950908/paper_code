import math
import numpy as np
import pyglet
from pyglet.gl import gl
from definitions import AgentType
RAD2DEG= 57.29577951308232
_GRID_COLOR=(0,0,0)
_BACKGROUND_COLOR=(255,255,255)
_SHELF_COLOR=(72,61,139)
_SHELF_PADDING=2
_AGENT_COLOR= (255,140,0)


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
    def render(self, env, return_rgb_array=False):
        gl.glClaerColor(*_BACKGROUND_COLOR,0)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        self._draw_grid
        self._draw_parking(env)
        self._draw_agents(env)
        self.window.flip()
        return self.isopen


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

    def _draw_parking(self,env):
        batch= pyglet.graphics.Batch()

        for parking in env.parking:
            x,y =parking[1], parking[0] # need debug
            y= self.rows-y-1 #pyglet rendering is reversed

            batch.add(
                4,
                gl.GL_QUADS,
                None,
                (
                    "v2f",
                    (
                        (self.grid_size+1)*x+_SHELF_PADDING +1, #top left x
                        (self.grid_size+1)*y+_SHELF_PADDING +1,  #top left y
                        (self.grid_size+1)*(x+1) - _SHELF_PADDING ,  # TL
                        (self.grid_size + 1) * y + _SHELF_PADDING + 1, #TR
                        (self.grid_size + 1) * (x + 1) - _SHELF_PADDING,  # BR - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  # BR - Y
                        (self.grid_size + 1) * x + _SHELF_PADDING + 1,  # BL - X
                        (self.grid_size + 1) * (y + 1) - _SHELF_PADDING,  #BL
                    ),
                ),
                ("c3B", 4*_SHELF_COLOR),
            )
        batch.draw()

    def _draw_agent(self,env):
        agent=[]
        batch = pyglet.graphics.Batch()

        radius = self. grid_size /3

        for agent in env.agents:
            col,row =agent[1],agent[0]
            row= self.rows-row-1

            if agent.type == AgentType.AGV:
                resolution=6

            verts = []
           
            for i in range(resolution):
                angle= 2 *math.pi*i/resolution
                x=(
                    radius*math.cos(angle)
                    +(self.grid_size+1)*col
                    +self.grid_size//2
                    +1
                )
                y=(
                    radius*math.sin(angle)
                    +(self.grid_si1+1)*row
                    +self.grid_size//2
                    +1
                )
                verts +=[x,y]
        circle= pyglet.graphics.vertex_list(resolution,("v2f",verts))
        draw_color= _AGENT_COLOR

        gl.glColor3ub(*draw_color)
        circle.draw(gl.GL_POLYGON) #pip install pyglet==1.5.21,this version support GL_POLYGON
        batch.draw()


