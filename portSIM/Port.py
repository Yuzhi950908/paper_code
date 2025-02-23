from portSIM.definitions import(Action,AgentType,RewardType,Duration)
from typing import Dict,List,Optional,Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import networkx as nx

#from portSIM.spaces import observation_map #它源文件其实是有一个 spaces子文件夹的，我根本不知道那是什么，有没有用，所以我没有写
#以下的建模，我个人觉得源代码对我没有用处的，我就没有写，然后部分逻辑是我自己写的

""" 以下先对车子进行建模 """

"""
核心目标
1. 初始化函数 ->  有ID,坐标,以及能赋予停车时长.
2. 功能 -> 可以由req_action 推导出下一步坐标.
"""

class Entity:
    def __init__(self,id_:int, x:int, y:int):
        self.id=id_
        self.x = x
        self.y = y

class Agent(Entity): 
    counter = 0
    
    def __init__(self, x: int, y: int, duration_type:Duration, agent_type: AgentType):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)
        self.duration = duration_type
        self.req_action: Optional[Action] = None
        self.type = agent_type
    
    def req_location(self,grid_size):  #->Tuple[int,int] 
        if self.req_action != Action.STAY:
            return self.x, self.y
        elif self.req_action == Action.UP:
            return self.x, max(0,self.y-1)
        elif self.req_action == Action.DOWN:
            return self.x, min(grid_size[0]-1, self.y+1)
        elif self.req_action == Action.LEFT:
            return max(0, self.x-1), self.y
        elif self.req_action == Action.RIGHT:
            return min(grid_size[1]-1,self.x+1),self.y
    
"""以下对env也就是 Port 进行建模"""

"""
核心目标
1. 输入 停车区域 的行数&列数， 以及对应的 每个区域的 长宽, 可以自动生成 停车场的layout参数
2. 输入小车的个数, 可以自动在第一排停车区域随机生成车子的坐标
3. 可以根据 外界生成并输入的goals(比如{'车1':'[5,3],[11,7],[14,2]'},生成所需的路径一系列坐标
4. 调用step,每个step,车子应当是可以任意改变当前的坐标的, 还得有终止step的功能
"""

class Port(gym.env):
    
    metadata= {"render_modes":["human","rgb_array"]}
    
    def __init__(self,parking_columns:int,column_height:int,parking_rows: int,row_width:int,num_agvs:int,goals:Tuple):

        self.goals = goals
        self.num_agvs= num_agvs
        self._make_layout_from_params(parking_columns,column_height,parking_rows,row_width)
        #self.observation 所有和oberservation 有关我都没写，不知道干嘛的
        self.renderer=None

    #下面实现第一个目标 make Layout,这个功能我已经验证过了。没有问题的
    def _make_layout_from_params(self,parking_columns,column_height,parking_rows,row_width):
        self._highway_lane=2
        self.column_width= row_width
        self.column_height= column_height
        self.grid_size=(
            self._highway_lane+(self.column_height+self._highway_lane)*parking_rows,#纵向
            self._highway_lane+(self.column_width+self._highway_lane)*parking_columns,
        )
        self.grid=np.zeros((self.grid_size),dtype=np.int32)
        self.grid_parking=np.zeros((self.grid_size),dtyoe=np.int32)
        self.grid_agvs=np.zeros((self.grid_size),dtype=np.int32)

        def get_highway_lanes_indices(axis_size,parking_size):
            return[
                i+j
                for i in range(0,axis_size,0+parking_size+self._highway_lane)
                for j in range(self._highway_lane)
            ]
        
        highway_ys= get_highway_lanes_indices(self.grid_size[0],self.column_height)
        highway_xs= get_highway_lanes_indices(self.grid_size[1],self.column_width)
        not_highway_ys=[num for num in  list(range(self.grid_size[0])) if num not in highway_ys]
        not_highway_xs=[num_ for num_ in  list(range(self.grid_size[1])) if num_ not in highway_xs]

        def fill_parking(not_highway_ys, not_highway_xs):
            parking_coord=[]
            for xs in not_highway_xs:
                for ys in not_highway_ys:
                    temp=[ys,xs]
                    parking_coord.append(temp)
            return parking_coord
        _parking_coord=fill_parking(not_highway_ys,not_highway_xs)

        for row,column in _parking_coord:
            self.grid_parking[row,column]=1
        
        return self.grid_parking
        # 到这里，我创造了一个Matrix，我把所有可停车区域 都赋值成了1，那么第一个功能实现,并且这里有一个特别重要的目的，
        # 这个self.grid_parking 可以直接接进 之后 找路的算法


    #下面实现第二个目标
    def fill_agvs(self):
        self._first_column_coord=[]
        for row in range(self.column_height):
            temp_first_row=[self._highway_lane + row, self._highway_lane]
            self._first_column_coord.append(temp_first_row)
            agent_loc_ids=np.random.choice(
                np.arange(len(self._first_column_coord)),
                size=self.num_agvs,
                replace=False,
            )
        self.agent_locs=[self._first_row_coord[i] for i in agent_loc_ids]
        #我在这里随机创造出来 5辆车子的初始位置，比如[array([5,2]),array([7,2]),array([4,2]),array([2,2]),array([3,2])]
        return self.agent_locs #这个就是下一个找路模块的 start_point

    #下面实现第三个目标，我换成了NetworkX  默认使用 BFS 算法      
    def path_finding(self,grid_parking,start_point,goals):
        G=nx.grid_2d_graph(len(grid_parking[0]), len(grid_parking[1]))
        for row in range(len(grid_parking[0])):
            for column in range(len(grid_parking[1])):
                if grid_parking[row][column] == 1:
                    G.remove_node((row, column))#我把之前定义成 1 的停车位拿掉，不准它走。只允许走highway。
        
        #然后我现在知道哪儿不可以走了。就可以定义 start point 和 goals了。
        for i in 

        


        
        
                 
