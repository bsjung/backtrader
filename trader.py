from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from datetime import datetime  # For datetime objects
import os.path  # To manage paths
import sys  # To find out the script name (in argv[0])

import backtrader as bt
from rl_strategy import RLStrategy


def btc_eth_300():
    currency_pair = "BTC_ETH"
    timeframe = 300
    file_name = currency_pair + "_" + str(timeframe) + "_small.csv"

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'history/' + file_name)

    return bt.feeds.GenericCSVData(
        dataname=datapath,
        name=currency_pair,
        headers=True,
        seperator=",",
        # timeframe=bt.TimeFrame.Seconds,
        # compression=timeframe,
        dtformat="%Y/%m/%d %H:%M:%S",
        fromdate=datetime(2017, 6, 1, 0, 0),
        todate=datetime.now(),
        volume=0,
        high=1,
        low=2,
        datetime=3,
        close=4,
        open=5,
        openinterest=-1
    )


if __name__ == '__main__':
    # Init and configure cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000)
    # Add a FixedSize sizer according to the stake
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)
    cerebro.broker.setcommission(commission=0.0)
    # Add a strategy
    cerebro.addstrategy(RLStrategy, n_bars=10)

    # Load data
    cerebro.adddata(btc_eth_300())

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    result = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    cerebro.plot()
