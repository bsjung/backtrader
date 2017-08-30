from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from random import randint
import backtrader as bt
import tensorflow as tf

'''
  Keep in mind:
  * Dont buy if another buy is in process
'''


# Create a Stratey
class RLStrategy(bt.Strategy):
    params = (
        ('timeframe', 5),
        ('n_bars', 15),  # bars used for analysing history
        ('data_per_bar', 4),  # bars used for analysing history
    )

    LEARNING_RATE = 0.001
    TF_INPUT_SIZE = 15 * 4
    TF_HIDDEN_1_SIZE = TF_INPUT_SIZE / 2
    TF_OUTPUT_SIZE = 2


    def log(self, txt, dt=None):
        ''' Logging function fot this strategy'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))


    def __init__(self):
        # Init Bactrader things:
        self.order = None
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.timeframe)
        # bt.indicators.WeightedMovingAverage(self.datas[0], period=25, subplot = True)
        # bt.indicators.SmoothedMovingAverage(rsi, period=10)
        # bt.indicators.ATR(self.datas[0], plot=False)

        # Init RL things:
        self.rl_init()

        # Init tensor things:
        self.tensor_init()


    ''' Methods for learning purposes: '''

    def rl_init(self):
        self.rl_reward = 0


    def rl_set_reward(self, reward):
        self.rl_reward = reward


    def action_sample(self):
        ''' executes and returns sample action '''
        ''' if no position is opened, then we return one of (buy, nothing) '''
        ''' if position is opened, then we return one of (sell, nothing) '''
        sample_number = randint(0, 50000) % 400

        # print("self.order: %s" % bool(self.order))

        # If no order present
        # and if not in market, we might open position:
        if not self.order and not self.position:
            if sample_number == 5:
                self.order = self.action_buy()
            else:
                self.action_nothing()
        elif not self.order and self.position:
            if sample_number == 5:
                self.action_close()
            else:
                self.action_nothing()


    def action_buy(self):
        ''' executes and returns buy action '''
        ''' only if no position is opened '''
        if not self.order:
            # print("return self.buy(): %d" % len(self))
            return self.buy()
        else:
            return self.order


    # def action_sell(self):
    #     ''' executes and returns sell action '''
    #     ''' only if position is opened '''
    #     print("return self.sell(): %d" % len(self))
    #     self.sell()
    #     return None


    def action_close(self):
        if self.position:
            return self.close()



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


    def get_reward(self):
        step_reward = self.rl_reward
        self.rl_reward = 0
        return step_reward



    ''' Tensor functions: '''

    def tensor_init(self):
        # We will pass as input array of flattened bar info
        self.tf_input = tf.placeholder(tf.float32, shape=[None, self.TF_INPUT_SIZE])
        # We will output probabily of taking action or wait.
        # If order is not present action => buy, otherwise => sell
        self.tf_output = tf.placeholder(tf.float32, shape=[None, 2])

        # Set up weights & biases of our model
        self.tf_w1 = tf.Variable(
            tf.random_normal([int(self.TF_INPUT_SIZE), int(self.TF_HIDDEN_1_SIZE)]))  # First weights
        self.tf_b1 = tf.Variable(tf.random_normal([int(self.TF_HIDDEN_1_SIZE)]))  # First biases
        self.tf_l1 = tf.nn.relu(tf.matmul(self.tf_input, self.tf_w1) + self.tf_b1)  # First layer

        self.tf_w2 = tf.Variable(
            tf.random_normal([int(self.TF_HIDDEN_1_SIZE), int(self.TF_OUTPUT_SIZE)]))  # Second weights
        self.tf_b2 = tf.Variable(tf.random_normal([int(self.TF_OUTPUT_SIZE)]))  # Second biases
        self.tf_result = tf.matmul(self.tf_l1, self.tf_w2) + self.tf_b2  # Resulting array

        self.tf_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_output, logits=self.tf_result)
        )
        self.tf_optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(self.tf_loss)



    def tensor_input(self):
        tensor_input = []
        for index in range(-self.params.n_bars+1, 1):
            tensor_input.append(self.tensor_input_entry(index))

        # flattens array of input:
        return [val for sublist in tensor_input for val in sublist]



    def tensor_input_entry(self, index):
        ''' One input entry for index '''
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
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            self.log('ORDER ACCEPTED/SUBMITTED')
            self.order = order
            return

        if order.status in [order.Expired]:
            self.log('BUY EXPIRED')

        elif order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

        # Sentinel to None: new orders allowed
        self.order = None


    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.rl_set_reward(trade.pnl)


        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))


    # Called on each bar
    def next(self):
        ''' We start doing things only after N  '''
        if len(self) <= self.params.n_bars:
            return

        # if len(self) == 300:
        #     print("positions:")
        #     print(self.position)
        #
        # if len(self) == 500:
        #     print("buy")
        #     self.buy()
        #
        # if len(self) == 578:
        #     print("positions:")
        #     print(self.position)
        #     print("action close:")
        #     self.action_close()
        #
        # if len(self) == 700:
        #     print("positions:")
        #     print(self.position)
        #     # self.action_sell()



        self.execute_action()