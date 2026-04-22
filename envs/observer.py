import gym
import numpy as np
from abc import abstractmethod, ABCMeta, ABC


class AbstractObserve(metaclass=ABCMeta):
    def __init__(self):
        super().__init__()
        self.observation = None
        self.init_observation = None
        self.observation_space = None

    def reset(self):
        self.observation[...] = self.init_observation

    def __call__(self, state, opponent):
        self.opponent = opponent
        if state is None:
            return self.init_observation
        else:
            return self.observe(state)

    @abstractmethod
    def observe(self, state):
        pass


class AbstractBinaryObserve(AbstractObserve, ABC):
    def __init__(self, domain, n_turn=2):
        super().__init__()
        self.domain = domain
        #self.maskは全ての選択肢を取得
        self.mask = [(i.name, j) for i in self.domain for j in i.values]
        # self.observation_space = gym.spaces.MultiBinary(len(self.mask) * n_turn)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(len(self.mask) * n_turn,), dtype=np.int8)
        self.my_offer = [0] * len(self.mask) * int((n_turn + 1) / 3)
        self.opp_offer1 = [0] * len(self.mask) * int((n_turn + 1) / 3)
        self.opp_offer2 = [0] * len(self.mask) * int((n_turn + 1) / 3)
        if n_turn == 1:
            self.init_observation = np.array(self.opp_offer1, dtype=np.int8)
        else:
            self.init_observation = np.array(self.my_offer + self.opp_offer1 + self.opp_offer2, dtype=np.int8)
        self.observation = np.empty_like(self.init_observation)


class OpponentObserve2n(AbstractObserve, ABC):
    def __init__(self, domain, n_turn=2):
        super().__init__()
        self.domain = domain
        self.mask = [(i.name, j) for i in self.domain for j in i.values]
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(n_turn, len(self.mask)), dtype=np.int8)
        self.init_observation = np.zeros((n_turn, len(self.mask)), dtype=np.int8)
        self.observation = np.empty_like(self.init_observation)

    def observe(self, state):
        offer = state['current_offer']
        if 'RLAgent' not in state['current_proposer']:
            self.observation[1:] = self.observation[:-1]
            self.observation[0, :] = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        return self.observation


class OpponentObserve1(AbstractObserve, ABC):
    def __init__(self, domain):
        super().__init__()
        self.domain = domain
        self.mask = [(i.name, j) for i in self.domain for j in i.values]
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(len(self.mask),), dtype=np.int8)
        self.init_observation = np.zeros((len(self.mask),), dtype=np.int8)
        self.observation = np.empty_like(self.init_observation)

    def reset(self):
        self.observation = np.zeros_like(self.init_observation)

    def observe(self, state):
        offer = state['current_offer']
        if 'RLAgent' not in state['current_proposer']:
            self.observation[:] = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        return self.observation


class OneHotObserve1(AbstractBinaryObserve):
    def __init__(self, domain):
        super().__init__(domain, n_turn=1)

    def observe(self, state):
        offer = state['current_offer']
        one_hot = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        if 'RLAgent' not in state['current_proposer']:
            self.opp_offer = one_hot
        self.observation = np.array(self.opp_offer, dtype=np.int8)
        return self.observation


class OneHotObserve2(AbstractBinaryObserve):
    def __init__(self, domain):
        super().__init__(domain)

    def observe(self, state):
        offer = state['current_offer']
        one_hot = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        if 'RLAgent' in state['current_proposer']:
            self.my_offer = one_hot
        else:
            self.opp_offer = one_hot
        self.observation = np.array(self.my_offer + self.opp_offer, dtype=np.int8)
        return self.observation


class OneHotObserve2n(AbstractBinaryObserve):
    def __init__(self, domain, n_turn):
        super().__init__(domain, n_turn)

    def observe(self, state):
        offer = state['current_offer']
        one_hot = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        if 'RLAgent' in state['current_proposer']:
            self.my_offer = one_hot + self.my_offer[:-len(self.mask)]
        else:
            self.opp_offer = one_hot + self.opp_offer[:-len(self.mask)]
        self.observation = np.array(self.my_offer + self.opp_offer, dtype=np.int8)
        return self.observation


# ここを使用するため、変更する
class OnehotObserve2nT(AbstractBinaryObserve):
    def __init__(self, domain, n_turn):
        super().__init__(domain, n_turn)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(len(self.mask)*n_turn+1,), dtype=np.float32)
        self.init_observation = np.array(self.my_offer + self.opp_offer1 + self.opp_offer2 + [0.], dtype=np.float32) #ここの最後の[0]は時間分
        self.observation = np.empty_like(self.init_observation)

    def observe(self, state):
        offer = state['current_offer'] 
        one_hot = [1 if offer[i[0]] == i[1] else 0 for i in self.mask]
        if state['current_proposer'] == 'RLAgent':
            self.my_offer = one_hot + self.my_offer[:-len(self.mask)]
        elif state['current_proposer'] == self.opponent[0]:
            self.opp_offer1 = one_hot + self.opp_offer1[:-len(self.mask)]
        elif state['current_proposer'] == self.opponent[1]:
            self.opp_offer2 = one_hot + self.opp_offer2[:-len(self.mask)]
        else:
            raise ValueError(f"Unknown proposer: {state['current_proposer']}")
        self.observation = np.array(self.my_offer + self.opp_offer1 + self.opp_offer2 + [state['relative_time']], dtype=np.float32) #時間を足している
        return self.observation


# class RLBOAObserve:
#     def __init__(self, domain, my_util):
#         self.domain = domain
#         self.my_util = my_util
#         self.observation_space = gym.spaces.Box(
#             np.array([0.0] * 5),
#             np.array([1.0] * 5),
#             dtype=np.float32
#         )
#         self.init_observation = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
#         self.observation = np.empty_like(self.init_observation)

#     def reset(self):
#         self.observation[...] = self.init_observation

#     def __call__(self, state):
#         if state is None:
#             return self.init_observation
#         else:
#             return self.observe(state)

#     def observe(self, state):
#         if 'RLAgent' in state['current_proposer']:
#             self.observation[2] = self.observation[0]
#             self.observation[0] = self.my_util(state['current_offer'])
#         else:
#             self.observation[3] = self.observation[1]
#             self.observation[1] = self.my_util(state['current_offer'])
#         self.observation[4] = state['relative_time']
#         return self.observation
