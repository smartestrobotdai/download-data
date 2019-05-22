from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pickle


class TradeStrategyFactory:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0.0, 0.005)},
                 {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.005, 0.0)},
                 {'name': 'stop_loss', 'type': 'continuous', 'domain': (-0.01,-0.003)},
                 {'name': 'stop_gain', 'type': 'continuous', 'domain': (0.002, 0.02)},
         ]
    def __init__(self, data, n_strategies=5, n_max_trades_per_day=4, slippage=0.00015, courtage=0):
        self.data = data
        self.n_max_trades_per_day = n_max_trades_per_day
        self.slippage = slippage
        self.courtage = courtage
        self.n_strategies = n_strategies
        self.trade_strategy_list = []
        self.trade_strategy = None
        return

    def create_trade_strategies(self, max_iter=200):
        init_numdata = int(max_iter / 4)
        for i in range(self.n_strategies):
            print("Searching Strategies, Run: {}".format(i))
            self.n_iter = 0
            self.trade_strategy = None
            self.max_profit = -1
            myBopt = GPyOpt.methods.BayesianOptimization(self.get_profit,  # Objective function       
                                                 domain=self.mixed_domain,          # Box-constraints of the problem
                                                 initial_design_numdata = init_numdata,   # Number data initial design
                                                 acquisition_type='EI',        # Expected Improvement
                                                 exact_feval = True,
                                                 maximize = True)           # True evaluations, no sample noise

            myBopt.run_optimization(max_iter, eps=0)

            self.trade_strategy_list.append(self.trade_strategy)

        return self.trade_strategy_list



    def get_profit(self, X_list):
        assert(len(X_list)==1)
        X_list = X_list[0]
        buy_threshold = X_list[0]
        sell_threshold = X_list[1]
        stop_loss = X_list[2]
        stop_gain = X_list[3]
        trade_strategy = TradeStrategy(X_list, self.n_max_trades_per_day, 
            self.slippage, self.courtage)
        total_profit, daily_profit_list =  trade_strategy.get_profit(self.data)
        avg_daily_profit = np.mean(daily_profit_list)
        if avg_daily_profit > self.max_profit:
            print("find new record: {}, {}".format(avg_daily_profit, 
                    trade_strategy.get_parameter_str()))

            self.max_profit = avg_daily_profit
            self.trade_strategy = trade_strategy

        self.n_iter += 1
        if self.n_iter % 50 == 0:
            print("iteration: {}, avg_daily_profit:{}".format(self.n_iter, avg_daily_profit))
        return np.mean(daily_profit_list).reshape((1,1))

def print_verbose_func(verbose, msg):
    if verbose == True:
        print(msg)


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


    def get_profit(self,  test_data, verbose=False):
        X_list = self.to_list()
        tot_profit, n_tot_trades, daily_profit_list, _, _ = self.run_test_core(X_list, 
                                                                                input_data, 
                                                                                verbose)
        return tot_profit, daily_profit_list


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

        reshaped_price = input_data[:,:,2].reshape((shape[0]*shape[1]))
        
        stock_change_rate = np.diff(reshaped_price) / reshaped_price[:-1]
        stock_change_rate = np.concatenate(([0], stock_change_rate)).reshape((shape[0],shape[1],1))
        
        asset_change_rate = np.zeros((stock_change_rate.shape))
        
        
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
                change_rate = stock_change_rate[day_idx][step][0]
                if state == 0 and time.time().hour >= 9 and \
                    n_trades < n_max_trades and \
                    step < len(daily_data)-5 and \
                    value > buy_threshold:
                        state = 1
                        asset_change_rate[day_idx][step][0] = -cost
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
                        print_verbose("sell at step: {} price:{} trade_profit:{} hold_steps:{}".format(step, price, trade_profit, hold_steps))
                        trade_profit = 1
                        asset_change_rate[day_idx][step] = change_rate
                        hold_steps = 0
                        
                    else:
                        tot_profit *= (1+change_rate)
                        daily_profit *= (1+change_rate)
                        trade_profit *= (1+change_rate)
                        asset_change_rate[day_idx][step][0] = change_rate
                        hold_steps += 1
            print_verbose("finished day {}, daily profit:{}".format(day_idx,daily_profit))
            daily_profit_list.append(daily_profit - 1)
            n_tot_trades += n_trades
        
        tot_profit -= 1
        return tot_profit, n_tot_trades, daily_profit_list, stock_change_rate, asset_change_rate
    
    def get_save_filename(self, path):
        return os.path.join(path, 'strategy_desc.pkl')
    
    def save(self, save_path):
        assert(self.trade_strategy_desc != None)
        with open(self.get_save_filename(save_path), 'wb') as f:
            pickle.dump(self.trade_strategy_desc, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, save_path):
        with open(self.get_save_filename(save_path), 'rb') as f:
            self.trade_strategy_desc = pickle.load(f)

if __name__ == '__main__':
    print("start testing")
    data = np.load("./npy_files/ema20_beta99_5.npy", allow_pickle=True)
    input_data = data[:60,6:-5,[-2,-3,-1]]
    trade_strategy_factory = TradeStrategyFactory(data, n_strategies=2)
    strategy_list = trade_strategy_factory.create_trade_strategies(max_iter=50)
    assert(len(strategy_list)==2)
    