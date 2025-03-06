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
    

    #这个req_location 之后可以手动操作汽车在格子里移动。
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


class Parkinglot(Entity):
    counter = 0

    def __init__(self, x, y):
        Parkinglot.counter +=1
        super().__init__(self.counter, x, y)


"""以下对env也就是 Port 进行建模"""

"""
核心目标
1. 输入 停车区域 的行数&列数， 以及对应的 每个区域的 长宽, 可以自动生成 停车场的layout参数
2. 输入小车的个数, 可以自动在第一排停车区域随机生成车子的坐标,这个坐标可以用于生成小车
3. 可以根据 外界生成并输入的goals(比如{'车1':'[5,3],[11,7],[14,2]'},生成所需的最短路径坐标
4. step 功能,这个里面功能应当是 包含了 reset 的功能. 
"""

"""
疑惑点
1. 怎么把DQL 比如observation reward 嵌进环境中
2. 怎么能渲染出图片
3. 怎么把这些模块接在一起运行起来
"""





class Port(gym.env):
    
    metadata= {"render_modes":["human","rgb_array"]}
    
    def __init__(self,parking_columns:int,column_height:int,parking_rows: int,row_width:int,num_agvs:int):
        self.num_agvs= num_agvs
        self.agvs=[]
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
            self._parking_coord=[]
            for xs in not_highway_xs:
                for ys in not_highway_ys:
                    temp=[ys,xs]
                    self._parking_coord.append(temp)
            return self._parking_coord
        self._parking_coord=fill_parking(not_highway_ys,not_highway_xs)

        for row,column in self._parking_coord:
            self.grid_parking[row,column]=1
        
        return self.grid_parking
        # 到这里，我创造了一个Matrix，我把所有可停车区域 都赋值成了1，那么第一个功能实现,并且这里有一个特别重要的目的，
        # 这个self.grid_parking 可以直接接进 之后 找路的算法









    #下面实现第二个目标
    def fill_agvs(self):
        self._first_column_coord=[]
        for row in range(5):
            temp_first_row=[self._highway_lane + row, self._highway_lane]
            self._first_column_coord.append(temp_first_row)
            agent_loc_ids=np.random.choice(
                np.arange(len(self._first_column_coord)),
                size=self.num_agvs,
                replace=False,
            )
        self.agent_locs_array=[self._first_row_coord[i] for i in agent_loc_ids]
        self.agent_locs=[arr.todolist() for arr in self.agent_locs_array]
        #我在这里随机创造出来 5辆车子的初始位置，比如[array([5,2]),array([7,2]),array([4,2]),array([2,2]),array([3,2])]
        return self.agent_locs #这个就是下一个找路模块的 start_point










    #下面实现第三个目标，我换成了NetworkX  默认使用 BFS 算法      
    def path_finding(self,grid_parking,start_point,goals):
        G=nx.grid_2d_graph(len(grid_parking[0]), len(grid_parking[1]))
        for row in range(len(grid_parking[0])):
            for column in range(len(grid_parking[1])):
                if grid_parking[row][column] == 1:
                    G.remove_node((row, column))#我把之前定义成 1 的停车位拿掉，不准它走。只允许走highway。
        shortest_path={}
        for i in range(len(self.agent_locs)):
            initial_loc=[self.agent_locs[i]]
            loc_list=goals[i].insert[0,initial_loc]
            path_finding=[]
            for index in range(len(loc_list)-1):#这个逻辑 主要是解决 终点再转运过程中会 变成起始点，比如小车到达2，5之后。 2，5变成起点了，11，7变成终点
                start_point=[loc_list[index]]
                end_point=[loc_list[index+1]]
                path = nx.shortest_path(G, source=start_point, target=end_point)
                path_finding.append(path)
        shortest_path[f'shortest_path for Agent{i}']=path_finding

        return shortest_path
    






    # 第四个目标: reset 和 step
    # reset: 应当是是可以重置汽车的位置和整个环境
    # step: 步数加一得同时，应当Agent的位置 都会随之更新
    def reset(self,seed=None, options=None):

        self._cur_steps = 0
        Parkinglot.counter = 0
        Agent.counter=0
        self.seed(seed)

        #先把 停车位实体创造出来, 我之前已经把停车场区域赋值成了1, highway是0了
        self.parking_lot=[
            Parkinglot(x,y)
            for y, x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if self.grid_parking[y,x]==1
        ]
        #然后确定Highway得坐标
        self._highway_locs= np.array([
            (y,x)
            for y,x in zip(
                np.indices(self.grid_size)[0].reshape(-1),
                np.indices(self.grid_size)[1].reshape(-1),
            )
            if self.grid_parking[y,x]==0
        ])

        #然后下一步是创造Agents 实体
        self.agent_locs=Port.fill_agvs()  #这里输出5个坐标

        self.agents= [
            Agent(x,y,Duration.SHORT,AgentType.AGV)
            for y,x in self.agent_locs           
        ]



    #这个step的难点就是，该能驱动Agent每一步都更新坐标
    def step(self,actions): #我之后输入一连串得shortest_Path做为actions
        self._cur_steps +=1
        done =False
        if not actions or self._cur_steps >=500:# 如果acitons是空集合了，那就结束，说明车子已经到了出口了。
            done= True
        
        else: #但凡不是空集，就把这个集合里的第一个元素提取出来给。这就是当前的车坐标
            self._cur_loc=actions.pop(0)
            #这里我暂时就想让第一辆车动起来就结束，我要更新这个第一个agent的位置就行了，其他4辆保持不动
            self.agents[0].y, self.agents[0].x = self._cur_loc
        
        #到这里结束。我因该只要运行一个step 我都能看到agents里车1 坐标在变化       
        return done
    
    def render(self,mode="human"):
        if not self.renderer:
            from portSIM.rendering import Viewer
            self.renderer= Viewer(self.grid_size)
        return self.renderer.render(self,return_rgb_array = mode == "rgb_array")

    def close(self):
        if self.renderer:
            self.renderer.close()



    





        


        
        
                 
