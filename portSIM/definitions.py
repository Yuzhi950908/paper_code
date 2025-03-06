from enum import Enum, IntEnum


class AgentType(Enum):
    AGV =0 #我们得逻辑里 单纯只有车子

class Duration(Enum): #新加入！ 区分长时停车 和 短时停车 和检修区
    SHORT=0
    LONG=1
    PDI=2


class Action(Enum): #我把Direction 删了，不需要看到 它有转向的动作，只要它能移动就行
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3  
    STAY=4  #静止


class RewardType(Enum): #我不知道这个之后有什么用
    GLOBAL=0
    INDIVIDUAL=1
    TWO_STAGE=2

class Layers(IntEnum):  
    AGVS=0
    PARKINGLOT=1
