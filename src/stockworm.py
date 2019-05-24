import numpy as np
from pathlib import Path
import pandas as pd
import uuid
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from util import remove_centralized

class StockWorm:
    def __init__(self, stock_name, stock_index, data_manipulator, model, 
      trade_strategy_model=None):

        self.stock_name = stock_name
        self.stock_index = stock_index    
        self.data_manipulator = data_manipulator
        self.model = model
        self.trade_strategy_model = trade_strategy_model

    # used for training.
    def get_profit(self, strategy_model_list, start_day_index=0, end_day_index=60):
        data_manipulator = self.data_manipulator
        model = self.model

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
        best_strategy_model = None
        assert(len(strategy_model_list)>0)
        for strategy_model in strategy_model_list:
          total_profit, profit_daily = strategy_model.get_profit(strategy_data_input)
          if total_profit > max_total_profit:
            max_total_profit = total_profit
            max_profit_daily = profit_daily
            best_strategy_model = strategy_model

        return max_total_profit, max_profit_daily, errors_daily, best_strategy_model
    

    def do_test(self, start_day_index, end_day_index):
        return
    
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
    



    def set_strategy_model(self, strategy_model):
        self.strategy_model = strategy_model


    
    def get_data_manipulator_filename(self, path):
        return os.path.join(path, 'data_manipulator.pkl')
    
    def get_strategy_model_filename(self, path):
        return os.path.join(path, 'data_manipulator.pkl')

    def save(self, path):
        # what is the last training date?
        self.model.save(path, self.data_manipulator.last_learning_date)
        
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
            daily_arr = seq_list
        else:
            shape = seq_list.shape
            n_days = int(shape[0] / 2)
            n_steps = shape[1] * 2
            n_columns = shape[2]
            daily_arr = seq_list.reshape((n_days, n_steps, n_columns))

        # remove the centralized bid part of data.
        # from 9:01 to 17:24
        if daily_arr.shape[1] == 516:
            daily_arr = remove_centralized(daily_arr)

        assert(daily_arr.shape[1]==504)
        return daily_arr