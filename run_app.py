import time
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
import numpy as np
import gymnasium as gym
"""
heuristic_episode:
这个整个run_app 文件都是伪代码, 跑不起来,单纯只是展示一下, 我想通过这个heuritstic_episode 到达什么目的.
目的是,我自己给他手动分配一个停车位,这个我在 portSIM.__init__文件的goals定义好了 。 然后要能显示出, 车子移动的坐标就行了。最后停在这个停车位就行了
"""

seed=0
def heuristic_episode(env,seed=None):
    #先初始化把整体停车场的环境创造出来
    env.reset(seed)
    #这里结束，我应该是能输出车的起点，以及网格参数

    #然后我应该生成一系列的路径坐标作为step接下来需要执行的actions。
    actions=env.path_finding(env.grid_parking,env.agent_locs[0],env.goals)
    #这里结束 我应该能得到从起始点开始，到我自己给他分配的停车位的一系列路径坐标

    #然后我把actions 输入step， 只要它actions都执行完了，变成空集合了。done 就是True了，一个episiode结束了
    done=env.step(actions)
    

    return infos #我的构想是 跑episode的同时，能够在信息里输出车子的坐标位置就行了，这里暂时不会写,不知道怎么完成
#这里结束我把我的逻辑呈现出来了。下面创造环境输入具体数据，开始跑模型






#先注册环境
env =gym.make("port-small-1agvs")
num_episodes=1
completed_episodes=0
for i in range(num_episodes):
    start=time.time()
    infos=heuristic_episode(env,seed)
    end=time.time()
    completed_episodes +=1


# 一个episode结束，车子应该是停在我给他分配好的停车位上的。

