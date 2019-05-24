import numpy as np
from sklearn import preprocessing
from util import remove_centralized
class TimeFormat:
    NONE = 0
    DAY = 1
    WEEK = 2

class DataManipulator:
    def __init__(self,  n_learning_days,
                n_prediction_days, beta, ema, time_format, volume_input, use_centralized_bid, 
                split_daily_data, stock_index):
        self.n_learning_days = n_learning_days
        self.n_prediction_days = n_prediction_days
        self.beta = beta
        self.ema = ema
        self.time_format = time_format
        self.volume_input = volume_input
        self.use_centralized_bid = use_centralized_bid
        self.split_daily_data = split_daily_data
        self.last_learning_date = None
        self.next_prediction_seq = None
        self.next_learning_seq = None
        self.stock_index = stock_index
        if split_daily_data == True:
            self.n_learning_seqs = self.n_learning_days * 2
            self.n_prediction_seqs = self.n_prediction_days * 2
        else:
            self.n_learning_seqs = self.n_learning_days
            self.n_prediction_seqs = self.n_prediction_days
        
        self.scaler_input = None
        self.scaler_output = None
    
    def update(self, next_prediction_seq, last_learning_date):
        assert(last_learning_date != None)
        print("updating, next_prediction_seq={}, last_learning_date={}".format(next_prediction_seq, last_learning_date))
        self.next_prediction_seq = next_prediction_seq
        self.next_learning_seq = next_prediction_seq - self.n_learning_seqs
        self.last_learning_date = last_learning_date
    
    def volume_transform(self, volume_series):
        # all the volumes must bigger than 0
        assert(np.all(volume_series>=0))
        return  np.log(volume_series.astype('float')+1)

    def inverse_transform_output(self, scaled_outputs):
        ori_shape = scaled_outputs.shape
        outputs_reshaped = scaled_outputs.reshape((ori_shape[0]*ori_shape[1], 
                                                   1))
        #outputs = np.exp(self.scaler_output.inverse_transform(outputs_reshaped)) - 1
        outputs = self.scaler_output.inverse_transform(outputs_reshaped)
        return outputs.reshape(ori_shape)
    
    def transform(self, data, n_inputs, n_outputs):
        input_scaled = self.transform_input(data[:,:,:n_inputs])
        output_scaled = self.transform_output(data[:,:,-n_outputs:])
        return input_scaled, output_scaled
    
    def transform_input(self, data_input):
        return self.transform_helper(self.scaler_input, data_input)
    
    def transform_output(self, data_output):
        return self.transform_helper(self.scaler_output, data_output)
        
    def transform_helper(self, scaler, data):
        shape = data.shape
        data = data.reshape(shape[0]*shape[1],shape[2])
        data_scaled = scaler.transform(data)
        return data_scaled.reshape(shape)
    
    # do fit and transform at same time
    def fit_transform(self, data_all, n_inputs, n_outputs):
        orig_shape = data_all.shape
        data_train_reshape = data_all.astype('float').reshape((orig_shape[0] * orig_shape[1], orig_shape[2]))
        
        self.scaler_input = preprocessing.MinMaxScaler().fit(data_train_reshape[:,:n_inputs])
        data_train_input_scaled = self.scaler_input.transform(data_train_reshape[:,:n_inputs])
        
        # the invalid step, we change it to zero!
        data_train_input_scaled[~np.any(data_train_reshape, axis=1)] = 0
        data_train_input = data_train_input_scaled.reshape(orig_shape[0], orig_shape[1], n_inputs)
        
        self.scaler_output = preprocessing.MinMaxScaler().fit(data_train_reshape[:,-n_outputs:])
        data_train_output_scaled = self.scaler_output.transform(data_train_reshape[:,-n_outputs:])
        # the invalid step, we change it to zero!
        data_train_output_scaled[~np.any(data_train_reshape, axis=1)] = 0
        data_train_output = data_train_output_scaled.reshape(orig_shape[0], orig_shape[1], n_outputs)
        
        return data_train_input, data_train_output

    # to purge data based on parameters like time_input, split_daily_data, etc.
    def purge_data(self, input_path, stock_index):
        # load numpy file
        npy_file_name = input_path + "/ema{}_beta{}_{}.npy".format(self.ema, self.beta, stock_index)
        input_np_data = np.load(npy_file_name, allow_pickle=True)
            
        # the diff is the mandatory
        input_columns = [2]
        
        time_format = self.time_format
        
        if time_format == TimeFormat.DAY:
            input_columns += [0]
        elif time_format == TimeFormat.WEEK:
            input_columns += [1]
        
        if self.volume_input == 1:
            input_columns += [3]
        
        output_columns = [4]
        timestamp_column = [5]
        price_column = [6]
        input_np_data = input_np_data[:,:,input_columns + output_columns + timestamp_column + price_column]
        
        # we must tranform the volume for it is too big.
        if self.volume_input == 1:
            input_np_data[:,:,-4] = self.volume_transform(input_np_data[:,:,-4])
        
        if self.use_centralized_bid == 0:
            # remove all the rows for centralized bid. it should be from 9.01 to 17.24, which is 516-12=504 steps
            input_np_data = remove_centralized(input_np_data)
            
            
        shape = input_np_data.shape
        if self.split_daily_data == 1:
            assert(shape[1] % 2 == 0)
            input_np_data = input_np_data.reshape((shape[0]*2, 
                                                  int(shape[1]/2), 
                                                  shape[2]))
            # get the first date and last date
        
        return input_np_data, input_columns
    
    def prep_training_data(self, start_day_index, end_day_index):
        input_path = 'npy_files'
        stock_index = self.stock_index
        input_np_data, input_columns = self.purge_data(input_path, stock_index)
        # to scale the data, but not the timestamp and price
        if self.split_daily_data == True:
            start = start_day_index * 2
            end = end_day_index * 2
        else:
            start = start_day_index
            end = end_day_index
            
        data_train_input, data_train_output = self.fit_transform(input_np_data[start:end,:,:-2], len(input_columns), 1)
        return data_train_input, data_train_output, input_np_data[start:end,:,-2], input_np_data[start:end,:,-1]
    
    def prep_testing_data(self, input_path, stock_index):
        input_path = 'npy_files'
        stock_index = self.stock_index
        input_np_data, n_training_sequences, input_columns = self.purge_data(input_path, stock_index)
        test_start_seq = self.next_prediction_seq - self.n_learning_seqs
        data_test_input, data_test_output = self.transform(input_np_data[test_start_seq:,:,:-2], len(input_columns), 1)
        return data_test_input, data_test_output, input_np_data[test_start_seq:,:,-2], input_np_data[test_start_seq:,:,-1]
    