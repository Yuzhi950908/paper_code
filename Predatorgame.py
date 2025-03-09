import numpy as np 
import cv2
from PIL import Image
import time
import pickle
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

SIZE = 10  #grid_size
EPISODES= 3000
SHOW_EVERY= 30 #每隔30局看一下过程



FOOD_REWARD= 25 #如果吃到实物就是25的奖励
ENEMY_PENALITY = 300 #如果遇到了敌人惩罚
MOVE_PENALITY = 1  # 移动扣分 

epsilon= 0.6    #60%概率去随机探索
EPS_DECAY= 0.9998  #探索应当逐步变小
DISCOUNT= 0.95   #这个就是遗忘率

q_table = None

d= {1:(255,0,0), #blue
    2:(0,255,0), #green
    3:(0,0,255)} #red   
PLAYER_N=1
FOOD_N=2
ENEMY_N=3


# 以上是定义了各种参数
###########################################################################
# 以下是写创造 整体的环境
class Cube:
    def __init__(self):
        self.x=np.random.randint(0,SIZE)
        self.y=np.random.randint(0,SIZE)

    def __str__(self): #输出实体坐标
        return f'{self.x},{self.y}'

    def __sub__(self,other):
        return(self.x-other.x,self.y-other.y) #这是一种类的特殊方法 叫双减法。 用player - food会自动调用
    def action(self,choise):
        if choise == 0:
            self.move(x=1, y=1)
        elif choise ==1:
            self.move(x=-1, y=1)
        elif choise ==2:
            self.move(x=1, y=1)
        elif choise ==3:
            self.move(x=-1,y=-1)
        pass
    def move(self,x=False,y=False):
        if not x:
            self.x += np.random.randint(-1,2)

        else: 
            self.x +=x

        if not y:
            self.y += np.random.randint(-1,2)
        else:
            self.y +=y
        
        if self.x < 0:
            self.x=0
        elif self.x >= SIZE:
            self.x=SIZE -1
        if self.y< 0:
            self.y=0
        elif self.y >= SIZE:
            self.y=SIZE -1
        
        





player=Cube()
print(player)
food=Cube()
player.action(0)
print(player)

    
