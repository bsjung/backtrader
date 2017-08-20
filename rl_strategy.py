from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

import backtrader as bt

# Create a Stratey
class RLStrategy(bt.Strategy):
  params = (
      ('timeframe', 5),
      ('bar_count', 100)
  )


  def log(self, txt, dt=None):
    ''' Logging function fot this strategy '''
    dt = dt or self.datas[0].datetime.date(0)
    print('%s, %s' % (dt.isoformat(), txt))


  def action_sample(self):
    ''' executes and returns sample action (buy, sell, nothing) '''
    pass


  def action_buy(self):
    ''' executes and returns buy action '''
    pass


  def action_sell(self):
    ''' executes and returns sell action '''
    pass


  def action_nothing(self):
    ''' executes and returns nothing action '''
    pass


  def start(self):
    self.log('Backtesting is about to start')


  def stop(self):
    self.log('Backtesting is finished')


  # Called on each bar
  def next(self):
    print(self.__dict__.keys())
    print("---")


