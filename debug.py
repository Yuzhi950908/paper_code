import numpy as np
import networkx as nx
np.random.seed(42)  # 固定随机种子

#先把基础环境构建好
###########################################################################################################################
class Port:
    def __init__(self,parking_columns:int,column_height:int,parking_rows: int,row_width:int,num_agvs:int):
        self.num_agvs= num_agvs
        self._make_layout_from_params(parking_columns,column_height,parking_rows,row_width)


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
        self.grid_parking=np.zeros((self.grid_size),dtype=np.int32)
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
        self.agent_locs=[self._first_column_coord[i] for i in agent_loc_ids]
        #我在这里随机创造出来 5辆车子的初始位置，
        return self.agent_locs #这个就是下一个找路模块的 start_point
    
    def path_finding(self,grid_parking,start_point,goals):
        rows, cols = len(grid_parking), len(grid_parking[0])
        G = nx.grid_2d_graph(rows, cols)
        start_point = tuple(start_point)
        goals = tuple(goals)
        grid_parking[start_point[0]][start_point[1]] = 0
        grid_parking[goals[0]][goals[1]] = 0
    # 移除停车位
        for row in range(rows):
            for col in range(cols):
                if grid_parking[row][col] == 1:
                    G.remove_node((row, col))
        shortest_path=nx.shortest_path(G,source=start_point,target=goals)
        return shortest_path


env = Port(
    parking_columns=5,  # 停车区域的列数
    column_height=6,  # 每列的高度
    parking_rows=1,    # 停车区域的行数
    row_width=1,       # 每行的宽度
    num_agvs=5         # AGV 的数量
)

print(env.grid_size)
print(env.grid_parking)
agent_init_locs=env.fill_agvs()
print(agent_init_locs)

"""
[[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 1 0 0 1 0 0 1 0 0 1 0 0 1 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]
[[2, 2], [3, 2], [7, 2], [4, 2], [6, 2]]
以上的代码代码我成功生成了 停车区域的 Grid形状
以及5车车子的初始位置 都在第一列的停车位中
"""
#############################################################################################################################################


















#下面生成实体agvs
##########################################################################################################################################

class Entity:
    def __init__(self,id_:int, x:int, y:int):
        self.id=id_
        self.x = x
        self.y = y

class Agent(Entity): 
    counter = 0
    
    def __init__(self, x: int, y: int):
        Agent.counter += 1
        super().__init__(Agent.counter, x, y)

agvs_dict = {}
for y,x in agent_init_locs:

    agv=Agent(x,y)
    agvs_dict[agv.id]={'x':agv.x, 'y':agv.y}

print(agvs_dict)
"""
{1: {'x': 2, 'y': 2}, 2: {'x': 2, 'y': 3}, 3: {'x': 2, 'y': 7}, 4: {'x': 2, 'y': 6}, 5: {'x': 2, 'y': 4}}
我在这里已经成功创造出车子的实例了
"""

##########################################################################################################################################################

















#下面的目标就是我手动随机给分配一个停车位 我要能看每个step， 比如车1能按照 最短路径前进，最短路径已经在Port类里定义过了已经
############################################################################################################################################################
shortest_path=env.path_finding(env.grid_parking,agent_init_locs[0],[3,14])
print(shortest_path)
"""
{1: {'x': 2, 'y': 2}, 2: {'x': 2, 'y': 3}, 3: {'x': 2, 'y': 7}, 4: {'x': 2, 'y': 4}, 5: {'x': 2, 'y': 6}}
[(2, 2), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (2, 15), (3, 15)]

到这里我找到了最短路径
"""
###########################################################################################################################################################################















#接下来我定义step(也就是下面的follow_path)，这个模块主要是每个step，都能看到车1，能按照预先计算好的最短路径，进行移动
######################################################################################################################################################################
import time
def follow_path(shortest_path, start_point, goal):
    current_position = start_point
    print(f"起始位置: {current_position}")

    for i, step in enumerate(shortest_path):
        time.sleep(0.5)  

        # 
        current_position = step
        print(f"步骤 {i + 1}: 车的位置 {current_position}")

        # 检查是否到达目标
        if current_position == goal:
            print("到达目标位置 [3, 15]，任务完成！")
            print("done")
            break



start_point=(agent_init_locs[0])#就先考虑以下第一个车，别的4辆暂时不考虑同时动
goal=(3,14)

# 调用函数，模拟车沿着路径行驶
follow_path(shortest_path, start_point, goal)

"""
起始位置: (2, 2)
步骤 1: 车的位置 (2, 2)
步骤 2: 车的位置 (1, 2)
步骤 3: 车的位置 (1, 3)
步骤 4: 车的位置 (1, 4)
步骤 5: 车的位置 (1, 5)
步骤 6: 车的位置 (1, 6)
步骤 7: 车的位置 (1, 7)
步骤 8: 车的位置 (1, 8)
步骤 9: 车的位置 (1, 9)
步骤 10: 车的位置 (1, 10)
步骤 11: 车的位置 (1, 11)
步骤 12: 车的位置 (1, 12)
步骤 13: 车的位置 (1, 13)
步骤 14: 车的位置 (1, 14)
步骤 15: 车的位置 (1, 15)
步骤 16: 车的位置 (2, 15)
步骤 17: 车的位置 (3, 15)
到达目标位置 [3, 15]，任务完成！
done"
"""
##########################################################################################################################################################