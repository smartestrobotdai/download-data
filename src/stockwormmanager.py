import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt
import uuid
import os.path
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from functools import partial
from tradestrategy import TradeStrategyFactory
from stockworm import StockWorm
from optimizeresult import OptimizeResult
from util import md5

class StockWormManager:
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
      {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
      {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4,5,6)},
      {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,30,40)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (2,5,10,20)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,3,5,10,20,30,40)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
      {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (1,)}
     ]

    mixed_domain_test = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
      {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
      {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
      {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
      {'name': 'learning_period', 'type': 'discrete', 'domain': (20,)},
      {'name': 'prediction_period', 'type': 'discrete', 'domain': (10,)},
      {'name': 'n_repeats', 'type': 'discrete', 'domain': (1,)},
      {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
      {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
      {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
      {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
      {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
     ]

    def __init__(self, stock_name, stock_index):
        self.stock_name = stock_name
        self.stock_index = stock_index

    def search_worms(self, strategy_cache_file, stock_worm_cache_file, start_day=0, end_day=60, 
        max_iter=300, is_test=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain

        self.optimize_result = OptimizeResult()
        if os.path.isfile(stock_worm_cache_file):
            self.optimize_result.load(stock_worm_cache_file)

        self.stock_worm_cache_file = stock_worm_cache_file

        trade_strategy_factory = TradeStrategyFactory()
        strategy_list = trade_strategy_factory.create_from_file(strategy_cache_file, 10)

        opt_func = partial(self.opt_func, strategy_list, start_day, end_day)

        opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                                     domain=mixed_domain,           # Box-constraints of the problem
                                     initial_design_numdata = 30,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True, 
                                     maximize = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)

    def opt_func(self, strategy_list, start_day, end_day, X_list):
        assert(len(X_list) == 1)

        features = X_list[0]
        print("starting test: {}".format(self.get_parameter_str(features)))  
        cached_result, index = self.optimize_result.find_result(features)
        if cached_result is not None:
            total_profit = cached_result[0]
            n_days = cached_result[1]
            profit_mean = cached_result[2]
            error_mean = cached_result[3]
            print("find from cache. skip...")
        else:
            stock_worm = self.build_stock_worm(features)
            total_profit, profit_daily, errors_daily, best_strategy_model = stock_worm.get_profit(strategy_list, start_day, end_day)
            stock_worm.set_strategy_model(best_strategy_model)
            n_days = len(profit_daily)
            profit_mean = np.mean(profit_daily)
            error_mean = np.mean(errors_daily)

            self.optimize_result.insert_result(features, [total_profit, n_days, profit_mean, error_mean])
            print("result saved to: {}".format(self.stock_worm_cache_file))
            self.optimize_result.save(self.stock_worm_cache_file)

        print("total_profit:{} in {} days, profit_mean:{} error:{} parameters:{}".format(total_profit, 
                                    n_days,
                                    profit_mean,
                                    error_mean,
                                    self.get_parameter_str(features)))

        return np.array(profit_mean).reshape((1,1))

    def create_worms_from_cache(self, strategy_cache_file, stock_worm_cache_file, n_number,
            start_day, end_day):
        optimize_result = OptimizeResult()
        optimize_result.load(stock_worm_cache_file)
        top_worms = optimize_result.get_best_results(n_number, by=-2)

        trade_strategy_factory = TradeStrategyFactory()
        strategy_list = trade_strategy_factory.create_from_file(strategy_cache_file, 5)

        assert(len(top_worms) == n_number)
        for i in range(n_number):
            features = top_worms[i, :13]


            new_worm = self.build_stock_worm(features)
            total_profit, profit_daily, errors_daily, best_strategy_model = stock_worm.get_profit(strategy_list, start_day, end_day)
            
            md5_str = self.get_md5_str(features)
            stock_worm.save(md5_str)

    def run_worms(self, stock_worm_cache_file, n_number, start_day, end_day=None):
        # load worms.
        optimize_result = OptimizeResult()
        optimize_result.load(stock_worm_cache_file)
        top_worms = optimize_result.get_best_results(n_number, by=-2)
        for i in range(n_number):
            features = top_worms[i, :13]
            new_worm = self.build_stock_worm(features)
            md5_str = self.get_md5_str(features)
            new_worm.load(md5_str)




    def get_md5_str(self, X):
        params_str = self.get_parameter_str(X)
        return md5(params_str)

    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str    

    def build_stock_worm(self, features):
        n_neurons = int(features[0])
        learning_rate = features[1]
        num_layers = int(features[2])
        rnn_type = int(features[3])
        learning_period = int(features[4])
        prediction_period = int(features[5])
        n_repeats = int(features[6])
        beta = int(features[7])
        ema = int(features[8])
        time_format = int(features[9])
        volume_input = int(features[10])
        use_centralized_bid = int(features[11])
        split_daily_data = int(features[12])

        data_manipulator = DataManipulator(learning_period,
                                           prediction_period,
                                           beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data, self.stock_index)
        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats)

        return StockWorm(self.stock_name, self.stock_index, data_manipulator, model)


if __name__ == '__main__':
    stock_worm_manager = StockWormManager('Nordel', 5)
    stock_worm_manager.search_worms("strategy_cache.txt", "worm_cache.txt", is_test=False)