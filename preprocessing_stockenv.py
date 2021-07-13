import gym
from gym import spaces
import numpy as np
import pandas as pd
from gym.utils import seeding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle


# initial_money = 1000
# cost = 0 # 0 for money and number for stock price


class StockEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'rende.modes':['human']}

    def __init__(self, df,initial_money =1000, time_unit =0):
        super(StockEnv,self).__init__()
        self.time_unit = time_unit
        self.df = df
        self.initial_money = initial_money

        # self.pre_state = previous_state

        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # discrete action space  0,1,2  0-hold 1-buy 2-sell
        # money when buying or no stock,  cost: price at last buying, close: current close price, macdh, rsi,... 6 kinds
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,))
        self.data = self.df.loc[self.time_unit, :] #extract data at time_unit
        self.terminal = False  # gym.env done
        # self.turbulence_threshold = turbulence_threshold
        self.cost =0
        self.shares = 0
        self.state = np.array([self.initial_money] + [self.cost] + [self.shares] + [self.data.Close] + [self.data.macd] + [self.data.cci]+[self.data.kdj]+ [self.data.Volume]) # array
        # self.obs = np.array([self.flag] + [self.data.Close] + [self.data.macd] + [self.data.cci]+[self.data.kdj]+ [self.data.Volume]) # actual input
        print(self.state)
        # store the total  money
        self.money_memory = [self.initial_money]
        self.reward_memory = []
        # self.reset()
        self._seed()
        self.action_list =[]
        # self.model_name=model_name

    def buy(self):
        if self.state[1] !=0:
            pass
        else:
            self.state[1] = self.state[3]
            self.state[2] = round(self.state[0]/self.state[3],3)
            self.state[0] = 0
            # print()

    def sell(self):
        if self.state[2] == 0:
            pass
        else:
            self.state[0] = round(self.state[2]*self.state[3], 3)
            self.state[1] = 0  # cost =0
            self.state[2] = 0 # shares =0


    def step(self, action):
        self.action_list.append(action)
        self.terminal = self.time_unit >= len(self.df)-1
        if self.terminal:
            plt.plot(self.money_memory, 'r')
            return self.state, self.reward, self.terminal, {}
        else:
            # print(pre_price)
            start_total_asset = self.state[0]+ self.state[2] * self.state[3]
            self.take_action(action)  # update state[0] and state[1] self.flag
            self.time_unit += 1
            # print(self.time_unit)
            self.data = self.df.loc[self.time_unit, :]
            # print(self.data)
            self.state = np.array(
                [self.state[0]] + [self.state[1]] + [self.state[2]] + [self.data.Close] + [self.data.macd] + [
                    self.data.cci] + [self.data.kdj] + [self.data.Volume])
            end_total_asset = self.state[0] + self.state[2] * self.state[3]
            self.money_memory.append(end_total_asset)
            self.reward = end_total_asset - start_total_asset
            self.reward_memory.append(self.reward)
            # self.rewards_memory.append(self.reward)

            # self.obs = np.array([self.flag]+[self.data.Close] + [self.data.macd] + [self.data.cci]+[self.data.kdj]+ [self.data.Volume])
        return self.state, self.reward, self.terminal, {}

    def take_action(self, action):
        if action == 1:
            self.buy()
        if action == 2:
            self.sell()
        if action == 0:
            pass

    def reset(self):
        self.money_memory = [self.initial_money]
        self.action_list =[]
        self.reward_memory = []
        self.time_unit = 0
        self.data = self.df.loc[self.time_unit, :]
        self.terminal = False
        self.state = np.array([self.initial_money]+[self.cost]+[self.shares]+[self.data.Close] + [self.data.macd] + [self.data.cci]+[self.data.kdj]+ [self.data.Volume])
        # self.obs = np.array([self.flag]+[self.data.Close] + [self.data.macd] + [self.data.cci]+[self.data.kdj] + [self.data.Volume])
        return self.state

    def render(self, mode ='human', close =False):
        return self.state

    def _seed(self, seed =None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

# test