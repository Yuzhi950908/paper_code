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
LEARNING_RATE=0.1

q_table = None

d= {1:(255,0,0), #blue
    2:(0,255,0), #green
    3:(0,0,255)} #red   
PLAYER_N=1
FOOD_N=2
ENEMY_N=3


# 以上是定义了各种参数
################################################################################################################################
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
#############################################################################################################################




player=Cube()
print(player)
food=Cube()
player.action(0)
print(player)
############################################################################################################################################
# Q-table initial  
if q_table is None:
    q_table={}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1,SIZE):
            for x2 in range(-SIZE+1,SIZE):
                for y2 in range(-SIZE+1,SIZE):
                    q_table[(x1,y1),(x2,y2)]=[np.random.uniform(-5,0) for i in range(4)]
else:
    with open(q_table,'rb') as f:
        pickle.load(q_table,f)
print(len(q_table))
##############################################################################################################################################


episode_rewards= []
for episode in range(EPISODES):
    player=Cube()
    food=Cube()
    enemy=Cube()
    episode_reward=0
    for i in range(200):
        obs = (player-food,player-enemy)
        if np.random.random()>epsilon:
            action= np.argmax(q_table[obs])
        else:
            action = np.random.randint(0,4)  
        print(player)
        print(obs)
        print(action)
        player.action(action)
        print("after action:")
        print(player)
        if player.x == food.x and player.y == food.y:
            reward=FOOD_REWARD
        elif player.x ==enemy.x and player.y == enemy.y:
            reward=-ENEMY_PENALITY
        else:
            reward=-MOVE_PENALITY
        print(f'reward:{reward}')
        
        #Update the Q _table
        current_q = q_table[obs][action]
        print(f'current_q:{current_q}')
        new_obs = (player-food,player-enemy)
        print(f'new_q:{new_obs}')
        max_future_q=np.max(q_table[new_obs])
        print(f'max_future_q:{max_future_q}')
        if reward == FOOD_REWARD:
            new_q=FOOD_REWARD
        new_q=(1-LEARNING_RATE)*current_q + LEARNING_RATE*(reward+DISCOUNT*max_future_q)
        print(f'new_q:{new_q}')
        q_table[obs][action]=new_q

        break
    break
#((7, 1), (-2, -4)) 这个就是一个状态 -> 相当于 去 Q-table 里最大的Q值，输出对应的 动作