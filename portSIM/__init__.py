import itertools
import gymnasium as gym
from portSIM.Port import RewardType

#_obs_types # 我不理解什么叫 Observation，暂时没有写
_sizes={
  'small': (1,5), #我们得逻辑目前是1 行 ，5列停车位， 每列6个格子
}

#_request_queues 不符合我们的逻辑，我们的要求序列，是有特定长度得。

#_perms = 不知道这个怎么用，暂时没有写

for size,obs_types,num_agvs,num_pickers in _perms:
    gym.register(
        id=f"port-{size}-{num_agvs}agvs",
        entry_point="port.port_model:Port",
        kwargs={
            'column_height':6,
            'parking_rows':_sizes[size][0],
            'parking_columns':_sizes[size][1],
            'num_agvs': num_agvs,
            'num_pickers': num_pickers,
            'request_queues': _request_queues[size],
            'max_inacitivity_steps': None,
            'max_step': 500,
            'reward_type': RewardType.INDIVIDUAL,
            'Observation_type': obs_types
            },
    )