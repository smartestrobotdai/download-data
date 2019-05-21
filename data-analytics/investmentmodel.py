import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt
import uuid
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from functools import partial

class InvestmentModel:
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': (20,30,40)},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (2,5,10,20)},
          {'name': 'n_repeats', 'type': 'discrete', 'domain': (3,5,10,20,30,40)},
          {'name': 'beta', 'type': 'discrete', 'domain': (99,)},
          {'name': 'ema', 'type': 'discrete', 'domain': (20,)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
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

    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str    

    def __init__(self, stock_name, stock_index):
        self.stock_name = stock_name
        self.stock_index = stock_index
        self.id = str(uuid.uuid1())
        self.save_path = "model_{}_{}".format(self.stock_name, self.id)
        self.model = None
        self.max_profit = -999.0
        return
    
    def optimize(self, strategy_model_list, start_day=0, end_day=60, max_iter=300, is_test=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain

        opt_func = partial(self.opt_func, strategy_model_list, start_day, end_day)

        opt_handler = GPyOpt.methods.BayesianOptimization(f=opt_func,  # Objective function       
                                     domain=mixed_domain,           # Box-constraints of the problem
                                     initial_design_numdata = 30,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True, 
                                     maximize = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)

    def opt_func(self, strategy_model_list, start_day, end_day, X_list):
        assert(len(X_list) == 1)

        features = X_list[0]
        print("starting test: {}".format(self.get_parameter_str(features)))        
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

        total_profit, profit_daily, errors_daily = self.get_profit(data_manipulator, model, strategy_model_list, start_day, end_day)
        print("total_profit:{} in {} days, error:{} parameters:{}".format(total_profit, 
                                    len(profit_daily),
                                    np.mean(errors_daily),
                                    self.get_parameter_str(features)))
        profit_per_day = total_profit/len(profit_daily)
        return np.array(profit_per_day).reshape((1,1))
    
    def get_data_manipulator_filename(self):
        return os.path.join(self.save_path, 'data_manipulator.pkl')
    
    def save(self):
        # what is the last training date?
        self.model.save(self.save_path, self.data_manipulator.last_learning_date)
        
        # save the data_manipulator
        filename = self.get_data_manipulator_filename()
        with open(filename, 'wb') as f:
            pickle.dump(self.data_manipulator, f, pickle.HIGHEST_PROTOCOL)
        
        # save the strategy model
        self.strategy_model.save(self.save_path)
    
    
    def get_latest_dir(self, save_path):
        all_subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
        max_time = 0
        for dirname in all_subdirs:
            fullname = os.path.join(save_path, dirname)
            time = os.path.getmtime(fullname)
            if time > max_time:
                max_time = time
                result = dirname
        return result

    def load(self, load_date=None):
        save_path = self.save_path
        # iterate the path, and find out the latest date as last_training_date
        self.model = StatefulLstmModel()
        
        # get the latest directory
        if load_date == None:
            load_date = self.get_latest_dir(self.save_path)
        
        print("Loading model for date: {}".format(load_date))
        self.model.load(self.save_path, load_date)
        
        # load data manipulator
        with open(self.get_data_manipulator_filename(), 'rb') as f:
            self.data_manipulator = pickle.load(f)
        
        # load strategy
        self.strategy_model = StrategyModel()
        self.strategy_model.load(self.save_path)
        print("Model loaded!")
    
    def get_date(self, timestamps, seq_no):
        return timestamps[seq_no][0].date().strftime("%y%m%d")

    def seq_to_daily(self, seq_list, split_daily_data):
        if split_daily_data == 0:
            return seq_list
        else:
            shape = seq_list.shape
            n_days = int(shape[0] / 2)
            n_steps = shape[1] * 2
            n_columns = shape[2]
            return seq_list.reshape((n_days, n_steps, n_columns))

    # used for training.
    def get_profit(self, data_manipulator, model, strategy_model_list, start_day_index, end_day_index):

        data_training_input, data_training_output, timestamps, price \
            = data_manipulator.prep_training_data(start_day_index, end_day_index)
        
        n_learning_seqs = data_manipulator.n_learning_seqs
        n_prediction_seqs = data_manipulator.n_prediction_seqs
        
        np_values, np_errors, next_prediction_seq = self.run_model(model, data_training_input, data_training_output, 
                                            n_learning_seqs, n_prediction_seqs)
        
        last_learning_date = self.get_date(timestamps, next_prediction_seq-1)
        data_manipulator.update(next_prediction_seq, last_learning_date)
       
        errors_daily = np.mean(np_errors, axis=1)
        assert(len(errors_daily) != 0)

        # find the best trade strategy.
        # prepare data for the strategy optimization, including timestamp, value, price.
        np_values = data_manipulator.inverse_transform_output(np_values)
        strategy_data_input = np.stack((timestamps[n_learning_seqs:], 
                                        np_values, 
                                        price[n_learning_seqs:]), axis=2)
        split_daily_data = data_manipulator.split_daily_data
        strategy_data_input = self.seq_to_daily(strategy_data_input, split_daily_data)

        max_total_profit = -1
        max_profit_daily = None
        assert(len(strategy_model_list)>0)
        for strategy_model in strategy_model_list:
          total_profit, profit_daily = strategy_model.run_test(strategy_data_input)
          if total_profit > max_total_profit:
            max_total_profit = total_profit
            max_profit_daily = profit_daily

        return max_total_profit, max_profit_daily, errors_daily
    
    
    # run the model, do learning and prediction at same time, 
    # this will be used for both training and testing.
    # at the test phase, we should do prediction first
    def run_model(self, model, data_input, data_output, n_learning_seqs, n_prediction_seqs):
        # get the date list.
        n_training_seqs = len(data_input)
        errors = None
        all_outputs = None
        n_tot_prediction_seqs = 0
        print("start training: training_seq:{}, learning_seq:{}, prediction_seq:{}".format(n_training_seqs, 
                                                                                           n_learning_seqs, 
                                                                                           n_prediction_seqs))
        for i in range(0, n_training_seqs-n_learning_seqs+1, n_prediction_seqs):
            learning_end = i + n_learning_seqs
            print("start training from seq:{} - seq:{}".format(i, learning_end-1))
            model.fit(data_input[i:learning_end], data_output[:learning_end], n_prediction_seqs)
            next_prediction_seq = learning_end
            prediction_end = min(learning_end+n_prediction_seqs, len(data_input))
            
            if prediction_end <= learning_end:
                break
            
            print("start predicting from seq:{} - seq:{}".format(learning_end, 
                                                                       prediction_end-1))
            
            outputs = model.predict_and_verify(data_input[learning_end:prediction_end], 
                                     data_output[learning_end:prediction_end])
            print("output.shape")
            print(outputs.shape)
            y = data_output[learning_end:prediction_end]
            # error is a 1-D array for the every day error
            error = np.square(outputs-y)
            
            n_tot_prediction_seqs += outputs.shape[0]
            if i == 0:
                all_outputs = outputs
                errors = error
            else:
                all_outputs = np.concatenate((all_outputs, outputs), axis=0)
                errors = np.concatenate((errors, error), axis=0)
        return np.squeeze(all_outputs), np.squeeze(errors), next_prediction_seq
    