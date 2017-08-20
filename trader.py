from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

import backtrader as bt
from rl_strategy import RLStrategy


CURRENCY_PAIR = "BTC_ETH"
TIMEFRAME = 300
FILE_NAME = CURRENCY_PAIR + "_" + str(TIMEFRAME) + ".csv"

if __name__ == '__main__':

  # Init and configure cerebro
  cerebro = bt.Cerebro()
  cerebro.broker.setcash(10000)
  # Add a FixedSize sizer according to the stake
  cerebro.addsizer(bt.sizers.FixedSize, stake=1)
  cerebro.broker.setcommission(commission=0.0)
  # Add a strategy
  cerebro.addstrategy(RLStrategy, bar_count=100)

  # Load data
  modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
  datapath = os.path.join(modpath, 'history/' + FILE_NAME)
  data = bt.feeds.GenericCSVData(
    dataname=datapath,
    name=CURRENCY_PAIR,
    headers=True,
    seperator=",",
    # timeframe=bt.TimeFrame.Seconds,
    # compression=TIMEFRAME,
    dtformat="%Y/%m/%d %H:%M:%S",
    fromdate=datetime(2017, 5, 1, 0, 0),
    todate=datetime.now(),
    volume=0,
    high=1,
    low=2,
    datetime=3,
    close=4,
    open=5,
    openinterest=-1
  )
  cerebro.adddata(data)


  print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
  result = cerebro.run()
  print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

  # cerebro.plot()