from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from random import randint

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

import backtrader as bt

'''
  Keep in mind:
  * Dont buy if another buy is in process
'''


# Create a Stratey
class RLStrategy(bt.Strategy):
    params = (
        ('timeframe', 5),
        ('n_bars', 15)  # bars used for analysing history
    )


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy '''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))


    def __init__(self):
        self.order = None
        self.reward = 0


    ''' Methods for learning purposes: '''

    def action_sample(self):
        ''' executes and returns sample action '''
        ''' if no position is opened, then we return one of (buy, nothing) '''
        ''' if position is opened, then we return one of (sell, nothing) '''
        sample_number = randint(0, 300) % 10

        # print("self.order: %s" % bool(self.order))

        # If position is open (buy_order) we might sell
        if self.order:
            if sample_number == 5:
                # print("if self.order: self.order = self.action_sell()")
                self.order = self.action_sell()
            else:
                # print("if self.order: self.action_nothing()")
                self.action_nothing()

        # If there is no position, we might buy
        # only if there are no other buy in action
        elif not self.order:
            if sample_number == 5:
                # print("elif not self.order: self.order = self.action_buy()")
                self.order = self.action_buy()
            else:
                # print("elif not self.order: self.action_nothing()")
                self.action_nothing()


    def action_buy(self):
        ''' executes and returns buy action '''
        ''' only if no position is opened '''
        if not self.order:
            # print("return self.buy(): %d" % len(self))
            return self.buy()
        else:
            return self.order


    def action_sell(self):
        ''' executes and returns sell action '''
        ''' only if position is opened '''
        # print("return self.sell(): %d" % len(self))
        self.sell()
        return None


    def action_nothing(self):
        ''' executes and returns nothing action '''
        return None


    def execute_action(self):
        ''' Execute action based on RL '''
        gready_action = True

        if gready_action:
            self.action_sample()
        else:
            # Take action on learned things
            pass


    def tensor_input(self):
        tensor_input = []
        for index in range(-self.params.n_bars+1, 1):
            tensor_input.append(self.tensor_input_entry(index))

        # flattens array of input:
        return [val for sublist in tensor_input for val in sublist]



    # One input entry for index
    def tensor_input_entry(self, index):
        return [
            self.data.close[index],
            self.data.low[index],
            self.data.high[index],
            self.data.open[index]
        ]


    ''' Backtrader callbacks: '''

    def start(self):
        # self.log('Backtesting is about to start')
        pass


    def stop(self):
        # self.log('Backtesting is finished')
        pass


    def notify_order(self, order):
        ''' Buy/Sell order submitted/accepted to/by broker - Nothing to do '''
        if order.status == order.Submitted:
            # self.log('Order Submitted')
            return

        if order.status == order.Accepted:
            # self.log('Order Accepted')
            return

        # If an order has been completed
        if order.status in [order.Completed]:
            # self.log('Order Completed')
            pass

        # Broker could reject order if not enougth cash
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            # self.log('Order Canceled/Margin/Rejected')
            pass

        # If order is completed delete it from strategy
        self.order = None


    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        # self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


    # Called on each bar
    def next(self):
        ''' We start doing things only after N  '''
        if len(self) <= self.params.n_bars:
            return

        print(self.tensor_input())

        # self.execute_action()
        return
