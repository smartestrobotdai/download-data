from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pickle
from optimizeresult import OptimizeResult
from util import remove_centralized



class TradeStrategy:
    def __init__(self, X_list, n_max_trades_per_day, slippage, courtage):
        self.buy_threshold = X_list[0]
        self.sell_threshold = X_list[1]
        self.stop_loss = X_list[2]
        self.stop_gain = X_list[3]
        self.slippage = slippage
        self.courtage = courtage
        self.n_max_trades_per_day = n_max_trades_per_day
        return

    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{} \
            stop_gain:{}".format(self.buy_threshold,
                                  self.sell_threshold,
                                  self.stop_loss,
                                  self.stop_gain)
        return s

    def to_list(self):
        return [self.buy_threshold, self.sell_threshold, self.stop_loss, self.stop_gain]


    def get_profit(self,  input_data, verbose=False):
        X_list = self.to_list()
        tot_profit, n_tot_trades, daily_profit_list, change_rate, is_hold = self.run_test_core(X_list, 
                                                                                input_data, 
                                                                                verbose)
        return tot_profit, daily_profit_list, change_rate, is_hold


    def run_test_core(self, X_list, input_data, verbose=False):
        print_verbose = partial(print_verbose_func, verbose)
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]

        tot_profit = 1
        tot_stock_profit = 1
        buy_step = None
        n_max_trades = self.n_max_trades_per_day
        cost = self.slippage/2 + self.courtage
        n_tot_trades = 0
        # to prepare the result data
        shape = input_data.shape
        price_data = input_data[:,:,2]
        reshaped_price = price_data.reshape((shape[0]*shape[1]))

        stock_change_rate = np.diff(reshaped_price) / reshaped_price[:-1]
        stock_change_rate = np.concatenate(([0], stock_change_rate)).reshape((shape[0],shape[1]))
        
        asset_change_rate = np.zeros((stock_change_rate.shape))
        is_hold = np.zeros((stock_change_rate.shape))
        
        daily_profit_list = []
        
        for day_idx in range(len(input_data)):
            print_verbose("starting day {}".format(day_idx))
            n_trades = 0
            daily_profit = 1
            trade_profit = 1
            state = 0
            daily_data = input_data[day_idx]
            hold_steps = 0

            for step in range(len(daily_data)):
                time = daily_data[step][0]
                value = daily_data[step][1]
                price = daily_data[step][2]
                change_rate = stock_change_rate[day_idx][step]
                if state == 0 and time.time().hour >= 9 and \
                    n_trades < n_max_trades and \
                    step < len(daily_data)-5 and \
                    value > buy_threshold:
                        state = 1
                        asset_change_rate[day_idx][step] = -cost
                        is_hold[day_idx][step] = 1
                        tot_profit *= (1-cost)
                        daily_profit *= (1-cost)
                        trade_profit *= (1-cost)
                        print_verbose("buy at step: {} price:{}".format(step, price))
                elif state == 1:
                    if value < sell_threshold  or \
                        step == len(daily_data)-1 or \
                        trade_profit-1 < stop_loss or \
                        trade_profit-1 > stop_gain:
                        # don't do more trade today!
                        if trade_profit-1 < stop_loss:
                            print_verbose("stop loss stop trading!")
                            n_trades = n_max_trades

                        change_rate = (1+change_rate)*(1-cost)-1 
                        tot_profit *= (1 + change_rate)
                        daily_profit *= (1 + change_rate)
                        state = 0
                        n_trades += 1
                        print_verbose("sell at step: {} price:{} trade_profit:{} hold_steps:{}".format(step, 
                            price, trade_profit, hold_steps))

                        trade_profit = 1
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps = 0
                        
                    else:
                        tot_profit *= (1+change_rate)
                        daily_profit *= (1+change_rate)
                        trade_profit *= (1+change_rate)
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps += 1
                        is_hold[day_idx][step] = 1

            print_verbose("finished day {}, daily profit:{}".format(day_idx,daily_profit))
            daily_profit_list.append(daily_profit - 1)
            n_tot_trades += n_trades
        
        tot_profit -= 1
        change_rate = np.stack((stock_change_rate, asset_change_rate), axis=2)
        return tot_profit, n_tot_trades, daily_profit_list, change_rate, is_hold
    
    def get_save_filename(self, path):
        return os.path.join(path, 'strategy_desc.pkl')




if __name__ == '__main__':
    print("start testing")
    # trade_strategy_factory = TradeStrategyFactory()
    # strategy_list = trade_strategy_factory.create_from_file("test.txt", 5)
    # assert(len(strategy_list)==5)

    data = np.load("../npy_files/ema20_beta99_5.npy", allow_pickle=True)
    input_data = data[:60,:,[-2,-3,-1]]
    input_data = remove_centralized(input_data)
    print(input_data.shape)
    trade_strategy_factory = TradeStrategyFactory(input_data, cache_file="strategy_cache.txt")
    strategy_list = trade_strategy_factory.create_trade_strategies(iter=5, max_iter=50)
    assert(len(strategy_list)==5)
    

