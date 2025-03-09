import itertools
import gymnasium as gym
from portSIM.Port import RewardType

#_obs_types # 我不理解什么叫 Observation，暂时没有写
_sizes={
  'small': (1,5), #我们得逻辑目前是1 行 ，5列停车位， 每列6个格子
}

#_perms = 不知道这个怎么用，暂时没有写
size='small'
num_agvs=1 #先用一个测试一下
goals=[2,5] #这里暂时手动分配一个停车位
gym.register(
    id=f"port-{size}-{num_agvs}agvs",
    entry_point="portSIM.Port:Port",
    kwargs={
            'parking_columns': _sizes[size][1],
            'column_height':6,
            'parking_rows':_sizes[size][0],
            'row_width': 1,
            'num_agvs':num_agvs,
            'goals':goals #这个goals 应当是 这样的Tuple结构 [[(2,5),(11,7),(14,2)]，[....],[...],[...],[...]]
            },
    )