from functools import partial
import numpy as np
import GPy
import GPyOpt
import os
import pickle

def print_verbose_func(verbose, msg):
    if verbose == True:
        print(msg)


class TradeStrategyDesc:
    def __init__(self,
                 X_list):
        self.buy_threshold = X_list[0]
        self.sell_threshold = X_list[1]
        self.stop_loss = X_list[2]
        self.stop_gain = X_list[3]
        
    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{} \
            stop_gain:{}".format(self.buy_threshold,
                                                  self.sell_threshold,
                                                  self.stop_loss,
                                                  self.stop_gain)
        return s
    
    
    def to_list(self):
        return [[self.buy_threshold, self.sell_threshold, self.stop_loss, self.stop_gain, 
                 self.min_hold_steps,
                 self.max_hold_steps]]



class StrategyModel:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0.0, 0.005)},
                 {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.005, 0.0)},
                 {'name': 'stop_loss', 'type': 'continuous', 'domain': (-0.01,-0.003)},
                 {'name': 'stop_gain', 'type': 'continuous', 'domain': (0.002, 0.02)},
         ]
    def __init__(self, n_max_trades_per_day=4, slippage=0.00015, courtage=0, max_iter=50):
        self.max_profit = -999.0
        self.strategy_desc = None
        self.optimize_data = None
        self.tot_profit = None
        self.slippage = slippage
        self.courtage = courtage
        self.max_iter = max_iter
        self.n_iter = 0
        self.n_max_trades_per_day = n_max_trades_per_day
        return
    
    # append the data for the optimization
    def append_data(self, data):
        if self.optimize_data is None:
            self.optimize_data = data
        else:
            self.optimize_data = np.concatenate((self.optimize_data, data), axis=0)

    def optimize(self, optimize_data):
        self.trade_strategy_desc = None
        self.max_profit = -999.0
        self.input_data = self.optimize_data
        self.n_iter = 0
        init_numdata = int(self.max_iter / 4)
        opt_func = partial(self.get_total_profit, optimize_data)
        myBopt = GPyOpt.methods.BayesianOptimization(opt_func,  # Objective function       
                                             domain=self.mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = init_numdata,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True,
                                             maximize = True)           # True evaluations, no sample noise

        myBopt.run_optimization(self.max_iter, eps=0)
        self.input_data = None
        return 
    
    def run_test(self, test_data):
        print("starting test: {}".format(self.trade_strategy_desc.get_parameter_str()))

        X_list = self.trade_strategy_desc.to_list()
        return self.get_total_profit(test_data, X_list)
    
    def get_total_profit(self, input_data, X_list):
        assert(len(X_list) == 1)
        X_list = X_list[0]
        tot_profit, n_tot_trades, daily_profit_list, _, _ = self.run_test_core(X_list, 
                                                                                     input_data, 
                                                                                     verbose=False)
        if tot_profit > self.max_profit:

            trade_strategy_desc = TradeStrategyDesc(X_list)

            print("iter:{} new record: tot_profit:{} in {} seqs, params:{}".format(self.n_iter,
                                                                    tot_profit,
                                                                    len(daily_profit_list),
                                                                    trade_strategy_desc.get_parameter_str()))
            self.max_profit = tot_profit
            self.trade_strategy_desc = trade_strategy_desc
        elif self.n_iter % 50 == 0:
            trade_strategy_desc = TradeStrategyDesc(X_list)
            print("iter:{} tot_profit:{} in {} seqs, params:{}".format(self.n_iter,
                                                                    tot_profit,
                                                                    len(daily_profit_list),
                                                                    trade_strategy_desc.get_parameter_str()))
        self.n_iter += 1
        return np.array(tot_profit).reshape((1,1))
    
    def get_training_seq_num(self):
        if self.split_daily_data == True:
            return self.ema_window * 4
        else:
            return self.ema_window * 2
    
    # the input data is in shape (days, steps, [timestamp, value, price])
    def get_profit_ema(self, X_list):
        assert(len(X_list)==1)
        X_list = X_list[0]
        n_training_seq_num = self.get_training_seq_num()
        input_data = self.input_data[-n_training_seq_num:]
        
        tot_profit, n_tot_trades, seq_profit_list, \
            stock_change_rate, asset_change_rate = self.run_test_core(X_list, input_data)
        
        assert(len(seq_profit_list) == n_training_seq_num)
        
        daily_profit_list = get_daily_profit_list(seq_profit_list, self.split_daily_data)
        profit_ema = get_ema(daily_profit_list, self.ema_window)
        
        if profit_ema > self.max_profit_ema:
            print("find best profit_ema_{}: {} tot_profit:{}".format(
                                                                            self.ema_window,
                                                                            profit_ema,
                                                                            tot_profit,
                                                                            self.ema_window))

            self.max_profit_ema = profit_ema
            
            self.change_rate = np.concatenate((input_data, 
                                              stock_change_rate,
                                              asset_change_rate), axis=2)
            self.trade_strategy_desc = TradeStrategyDesc(X_list,
                                             self.ema_window,
                                             self.optimize_data)
            self.tot_profit = tot_profit
            self.max_profit_list = seq_profit_list
        
        return np.array(profit_ema).reshape((1,1))
    
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
    
    def get_max_profit_list(self):
        return self.max_profit_list
    
    def get_strategy_desc(self):
        return self.trade_strategy_desc
    
    def get_save_filename(self, path):
        return os.path.join(path, 'strategy_desc.pkl')
    
    def save(self, save_path):
        assert(self.trade_strategy_desc != None)
        with open(self.get_save_filename(save_path), 'wb') as f:
            pickle.dump(self.trade_strategy_desc, f, pickle.HIGHEST_PROTOCOL)
            
    def load(self, save_path):
        with open(self.get_save_filename(save_path), 'rb') as f:
            self.trade_strategy_desc = pickle.load(f)

