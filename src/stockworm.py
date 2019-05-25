import numpy as np
from pathlib import Path
import pandas as pd
import uuid
import os.path
import pickle
from datamanipulator import DataManipulator
from statefullstmmodel import StatefulLstmModel
from util import remove_centralized, timestamp2date

class StockWorm:
    def __init__(self, stock_index, input_data_path, save_path):
        self.stock_index = stock_index 
        self.input_data_path = input_data_path
        self.save_path = save_path
        self.create_if_not_exist(save_path)

        self.last_learning_day_index = None
        self.learning_end_date = None

    def create_if_not_exist(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)


    def init(self, features, strategy_model_list, start_day_index, end_day_index):
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

        data_manipulator = DataManipulator(self.stock_index,
                                           learning_period,
                                           prediction_period,
                                           beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data, 
                                           self.input_data_path)
        
        data_manipulator.init_scalers(start_day_index, end_day_index)

        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats)
        self.data_manipulator = data_manipulator
        self.model = model

        strategy_data_input, errors_daily = self.test_model_base(start_day_index, end_day_index)

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

        self.strategy_model = best_strategy_model
        return max_total_profit, max_profit_daily, errors_daily
    
    
    def test(self, end_day_index=None):
        # find the start_day_index
        assert(self.learning_end_date != None)
        n_learning_days = self.data_manipulator.get_learning_days()
        n_prediction_days = self.data_manipulator.get_prediction_days()
        learning_end_day_index = self.data_manipulator.date_2_day_index(self.learning_end_date)



        start_day_index = learning_end_day_index + 1 - n_learning_days + n_prediction_days

        if end_day_index is None:
            end_day_index = self.data_manipulator.get_n_days()

        strategy_data_input, errors_daily = self.test_model_base(start_day_index, end_day_index)
        total_profit, profit_daily = self.strategy_model.get_profit(strategy_data_input)

        return total_profit, profit_daily, errors_daily

    def test_model_base(self, start_day_index, end_day_index):
        data_manipulator = self.data_manipulator

        data_input, data_output, timestamps, price \
            = data_manipulator.prep_data(start_day_index, end_day_index)

        start_date = timestamp2date(timestamps[0][0])
        end_date = timestamp2date(timestamps[-1][0])
        n_days = end_day_index - start_day_index
        n_seqs = len(data_input)
        print("Running model from {} to {}, {} days, {} seqs".format(start_date, end_date, n_days, n_seqs))

        np_values, np_errors, learning_end_seq = self.run_model(data_input, data_output)
        
        
        self.learning_end_date = timestamp2date(timestamps[learning_end_seq][0])
        print("Running model finished, learning end date is:  {}".format(self.learning_end_date))

        errors_daily = np.mean(np_errors, axis=1)
        assert(len(errors_daily) == len(np_errors))

        # find the best trade strategy.
        # prepare data for the strategy optimization, including timestamp, value, price.

        np_values = data_manipulator.inverse_transform_output(np_values)
        n_learning_seqs = data_manipulator.get_learning_seqs()

        length = len(np_values)
        strategy_data_input = np.stack((timestamps[-length:], 
                                        np_values, 
                                        price[-length:]), axis=2)

        strategy_data_input = data_manipulator.seq_data_2_daily_data(strategy_data_input, is_remove_centralized=True)
        return strategy_data_input, errors_daily

    # run the model, do learning and prediction at same time, 
    # this will be used for both training and testing.
    # at the test phase, we should do prediction first
    def run_model(self, data_input, data_output):
        # get the date list.
        n_training_seqs = len(data_input)
        errors = None
        all_outputs = None

        n_learning_seqs = self.data_manipulator.get_learning_seqs()
        n_prediction_seqs = self.data_manipulator.get_prediction_seqs()

        # if the model is ready, this is where we should start to predict.
        learning_end = n_learning_seqs - n_prediction_seqs
        for i in range(0, n_training_seqs-n_learning_seqs+1, n_prediction_seqs):
            # the model is ready, we want to do prediction first.
            if self.model.is_initialized() == True:
                # do prediction first
                prediction_start = learning_end
                prediction_end = min(prediction_start+n_prediction_seqs, len(data_input))

                print("starting prediction from seq:{} to seq:{}".format(prediction_start, prediction_end-1))
                outputs = self.model.predict_and_verify(data_input[prediction_start:prediction_end], 
                            data_output[prediction_start:prediction_end])

                y = data_output[prediction_start:prediction_end]
                error = np.square(outputs-y)
                if all_outputs is None:
                    all_outputs = outputs
                    errors = error
                else:
                    all_outputs = np.concatenate((all_outputs, outputs), axis=0)
                    errors = np.concatenate((errors, error), axis=0)

            learning_end = i + n_learning_seqs
            print("start training from seq:{} - seq:{}".format(i, learning_end-1))
            self.model.fit(data_input[i:learning_end], data_output[:learning_end], n_prediction_seqs)
            self.initialized = True
 
        return np.squeeze(all_outputs), np.squeeze(errors), learning_end-1

    
    def get_data_manipulator_filename(self, path):
        return os.path.join(path, 'data_manipulator.pkl')
    
    def get_strategy_model_filename(self, path):
        return os.path.join(path, 'strategy.pkl')

    def save(self):
        assert(self.learning_end_date != None)
        path = self.save_path
        # what is the last training date?
        self.model.save(path, self.learning_end_date)
        
        # save the data_manipulator
        filename = self.get_data_manipulator_filename(path)
        with open(filename, 'wb') as f:
            pickle.dump(self.data_manipulator, f, pickle.HIGHEST_PROTOCOL)

        # save the strategy model
        filename = self.get_strategy_model_filename(path)
        with open(filename, 'wb') as f:
            pickle.dump(self.strategy_model, f, pickle.HIGHEST_PROTOCOL)
    
    
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
        path = self.save_path
        # iterate the path, and find out the latest date as last_training_date
        self.model = StatefulLstmModel()
        
        # get the latest directory
        if load_date == None:
            load_date = self.get_latest_dir(path)
        
        print("Loading model for date: {} under: {}".format(load_date, path))
        self.model.load(path, load_date)

        with open(self.get_data_manipulator_filename(path), 'rb') as f:
            self.data_manipulator = pickle.load(f)

        with open(self.get_strategy_model_filename(path), 'rb') as f:
            self.strategy_model = pickle.load(f)

        # recover the learning_end_date
        self.learning_end_date = load_date

if __name__ == '__main__':
    from tradestrategy import TradeStrategyFactory
    trade_strategy_factory = TradeStrategyFactory()
    strategy_list = trade_strategy_factory.create_from_file('strategy_cache.txt', 10)
    stock_worm = StockWorm(5, 'npy_files', 'my_model')

    features=[60.0 , 0.004 , 1.0 , 0.0 , 20.0 , 20.0 ,  1.0 , 99.0 , 20.0,  1.0,  1.0 , 1.0 , 1.0]
    total_profit, profit_daily, errors_daily = stock_worm.init(features, strategy_list, 0, 60)
    print("Training finished: total_profit:{}, profit_daily:{}".format(total_profit, profit_daily))
    stock_worm.save()

    total_profit1, profit_daily, errors_daily = stock_worm.test()
    print("Testing finished: total_profit:{}, profit_daily:{}".format(total_profit1, profit_daily))

    stock_worm2 = StockWorm(5, 'npy_files', 'my_model')
    stock_worm2.load()    

    total_profit2, profit_daily, errors_daily = stock_worm2.test()
    print("Testing finished: total_profit:{}, profit_daily:{}".format(total_profit2, profit_daily))
    stock_worm2.save()
    assert(total_profit1 == total_profit2)

    stock_worm3 = StockWorm(5, 'npy_files', 'my_model')
    stock_worm3.load('190423')
    total_profit3, profit_daily, errors_daily = stock_worm3.test()
    assert(total_profit1 == total_profit3)


