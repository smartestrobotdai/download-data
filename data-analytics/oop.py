#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import copy
from sklearn import preprocessing
import datetime
import pickle


# In[3]:


def sma(data, window):
    """
    Calculates Simple Moving Average
    http://fxtrade.oanda.com/learn/forex-indicators/simple-moving-average
    """
    if len(data) < window:
        return None
    return sum(data[-window:]) / float(window)

def get_ema(data, window):
    if len(data) < 2 * window:
        raise ValueError("data is too short")
    c = 2.0 / (window + 1)
    current_ema = sma(data[-window*2:-window], window)
    for value in data[-window:]:
        current_ema = (c * value) + ((1 - c) * current_ema)
    return current_ema


# In[4]:


class NetAttributes:
    def __init__(self, n_neurons = 100, 
                 learning_rate = 0.003, 
                 num_layers = 1,
                 rnn_type = 2,
                 n_repeats = 2):
        self.n_neurons = n_neurons;
        self.learning_rate = learning_rate;
        self.num_layers = num_layers;
        self.rnn_type = rnn_type;
        self.n_repeats = n_repeats
        self.n_steps = None
        self.n_inputs = None
        self.n_outputs = 1
        
    def set_input_dimension(self, n_steps, n_inputs):
        self.n_steps = n_steps
        self.n_inputs = n_inputs


# In[5]:


class NetStates:
    def __init__(self):
        self.prediction_states = None
        self.training_states = None
    


# In[6]:


class StatefulLstmModel:
    def __init__(self,
                n_neurons=100,
                learning_rate=0.002,
                num_layers=2,
                rnn_type=1,
                n_repeats=30):

        self.net_attributes = NetAttributes(n_neurons,
                                   learning_rate,
                                   num_layers,
                                   rnn_type,
                                   n_repeats)
        self.net_states = NetStates()
        self.model_initialized = False
        self.sess = None
    
    def __del__(self):
        if self.sess != None:
            self.sess.close()
    
    def get_batch(self, seq_index, data_train_input, data_train_output):
        X_batch = data_train_input[seq_index:seq_index+1]
        y_batch = data_train_output[seq_index:seq_index+1]
        return X_batch, y_batch
    
    
    def initialize_layers(self):
        layers = None
        net_attributes = self.net_attributes
        if net_attributes.rnn_type == 0:
            layers = [tf.nn.rnn_cell.BasicLSTMCell(net_attributes.n_neurons) 
              for _ in range(net_attributes.num_layers)]
        elif net_attributes.rnn_type == 1:
            layers = [tf.nn.rnn_cell.LSTMCell(net_attributes.n_neurons, use_peepholes=False) 
              for _ in range(net_attributes.num_layers)]
        elif net_attributes.rnn_type == 2:
            layers = [tf.nn.rnn_cell.LSTMCell(net_attributes.n_neurons, use_peepholes=True) 
              for _ in range(net_attributes.num_layers)]
        else:
            print("WRONG")
        return layers
    
    def reset_graph(self, seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
    
    def create_model(self):
        net_attributes = self.net_attributes
        self.X = tf.placeholder(tf.float32, [None, net_attributes.n_steps, net_attributes.n_inputs])
        self.y = tf.placeholder(tf.float32, [None, net_attributes.n_steps, net_attributes.n_outputs])
        layers = self.initialize_layers()
        cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        self.init_state = tf.placeholder(tf.float32, [net_attributes.num_layers, 2, 1, net_attributes.n_neurons])
        
        state_per_layer_list = tf.unstack(self.init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(net_attributes.num_layers)]
        )
        
        rnn_outputs, self.new_states = tf.nn.dynamic_rnn(cell, self.X, dtype=tf.float32, 
                                                    initial_state=rnn_tuple_state)
        
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, net_attributes.n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, net_attributes.n_outputs)
        self.outputs = tf.reshape(stacked_outputs, [-1, net_attributes.n_steps, net_attributes.n_outputs])
        
        self.loss = tf.reduce_mean(tf.square(self.outputs - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=net_attributes.learning_rate)
        self.training_op = optimizer.minimize(self.loss)

        self.init = tf.global_variables_initializer()
        self.model_initialized = True
    
    # train the model, input is the training data for one cycle
    # input is in the shape: [days, steps, features], the features are 
    # 1. diff, 2. volume. 3. timesteps.
    def fit(self, data_train_input, data_train_output, prediction_period):
        net_attributes = self.net_attributes
        net_states = self.net_states
        n_inputs = data_train_input.shape[2]
        n_steps = data_train_input.shape[1]

        net_attributes.set_input_dimension(n_steps, n_inputs)
        batch_size = 1
        days = data_train_input.shape[0]
        
        self.reset_graph()
        self.create_model()
        my_loss_train_list = []
        sess = tf.Session()
        # TODO: load from file.

        self.init.run(session=sess)
        # if this is the first time of fit?
        if self.net_states.training_states == None:
            init_states = np.zeros((net_attributes.num_layers, 2, 1, net_attributes.n_neurons))
        else:
            init_states = self.net_states.training_states
            
        for repeat in range(net_attributes.n_repeats):
            rnn_states = copy.deepcopy(init_states)
            for seq in range(days):
                X_batch, y_batch = self.get_batch(seq, data_train_input, data_train_output)
                feed_dict = {
                        self.X: X_batch,
                        self.y: y_batch,
                        self.init_state: rnn_states}
                my_op, rnn_states, my_loss_train, my_outputs = sess.run([self.training_op, 
                          self.new_states, 
                          self.loss, 
                          self.outputs], feed_dict=feed_dict)

                my_loss_train_list.append(my_loss_train)
                # last repeat , remember the sates
                if seq+1 == prediction_period and repeat == net_attributes.n_repeats-1:
                    # next training loop starts from here
                    training_states = copy.deepcopy(rnn_states)
                my_loss_train_avg = sum(my_loss_train_list) / len(my_loss_train_list)

            print("{} repeat={} training finished, training MSE={}".format(
                datetime.datetime.now().time(),
                repeat, my_loss_train_avg))
        
        self.net_states.training_states = training_states
        self.net_states.prediction_states = rnn_states
        self.sess = sess
        return
    
    def predict_base(self, data_test_input, data_test_output=None):
        net_attributes = self.net_attributes
        net_states = self.net_states
        days = data_test_input.shape[0]
        
        rnn_states = copy.deepcopy(net_states.prediction_states)
        #X, y, init_state, init, training_op, new_states, loss, outputs = self.create_model()
        sess = self.sess
        
        my_loss_test_list = []
        input_shape = data_test_input.shape
        outputs_all_days = np.zeros((input_shape[0], input_shape[1], 1))
        for seq in range(days):
            if data_test_output is None:
                feed_dict = {
                    self.X: data_test_input[seq:seq+1],
                    self.init_state: rnn_states,
                }

                rnn_states, my_outputs = sess.run([self.new_states, self.outputs], feed_dict=feed_dict)
            else:
                feed_dict = {
                    self.X: data_test_input[seq:seq+1],
                    self.y: data_test_output[seq:seq+1],
                    self.init_state: rnn_states,
                }

                rnn_states, my_outputs, my_loss_test = sess.run([self.new_states, 
                                                                 self.outputs, self.loss], feed_dict=feed_dict)
                print("Predicting seq:{} testing MSE: {}".format(seq, my_loss_test))
            outputs_all_days[seq] = my_outputs
            
        
        return outputs_all_days
    
    def predict(self, data_test_input):
        return self.predict_base(data_test_input)
        
    def predict_and_verify(self, data_test_input, data_test_output):
        return self.predict_base(data_test_input, data_test_output)
      
    def get_attributes_filename(self, path):
        if path[-1] != '/':
            path += '/'
        return path + 'net_attributes.pkl'
    
    def get_path(self, path, date):
        if path[-1] != '/':
            path += '/'
        return path + date + '/'
    
    def get_states_filename(self, path, date):
        return self.get_path(path, date) + 'net_states.pkl'
    
    def get_model_filename(self, path, date):
        return self.get_path(path, date) + '/tf_session.ckpt'
    
    def save(self, path, date):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, self.get_model_filename(path, date))
        with open(self.get_attributes_filename(path), 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(self.net_attributes, f, pickle.HIGHEST_PROTOCOL)
        with open(self.get_states_filename(path, date), 'wb') as f:
            pickle.dump(self.net_states, f, pickle.HIGHEST_PROTOCOL)
        print("Model saved in path: %s" % path)
        
            
    def load(self, path, date):
        # TODO: if date is none, load the latest.
        
        # restore hyper-params
        with open(self.get_attributes_filename(path), 'rb') as f:
            self.net_attributes = pickle.load(f)

        # restore states
        with open(self.get_states_filename(path), 'rb') as f:
            self.net_states = pickle.load(f)
        
        # 2. restore graph
        if self.model_initialized == False:
            self.reset_graph()
            self.create_model()
        
        # 3. restore session
        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, self.get_model_filename(path))
        print("Model restored.")


# In[7]:


class TimeFormat:
    NONE = 0
    DAY = 1
    WEEK = 2

class DataManipulator:
    def __init__(self, beta, ema, time_format, volume_input, use_centralized_bid, 
                split_daily_data, n_training_days):
        self.beta = beta
        self.ema = ema
        self.time_format = time_format
        self.volume_input = volume_input
        self.use_centralized_bid = use_centralized_bid
        self.split_daily_data = split_daily_data
        self.n_training_days = n_training_days
        self.scaler_input = None
        self.scaler_output = None
        
    def volume_transform(self, volume_series):
        # all the volumes must bigger than 0
        assert(np.all(volume_series>=0))
        return  np.log(volume_series.astype('float')+1)

    def inverse_transform_output(self, scaled_outputs):
        ori_shape = scaled_outputs.shape
        outputs_reshaped = scaled_outputs.reshape((ori_shape[0]*ori_shape[1], 
                                                   ori_shape[2]))
        #outputs = np.exp(self.scaler_output.inverse_transform(outputs_reshaped)) - 1
        outputs = self.scaler_output.inverse_transform(outputs_reshaped)
        return outputs.reshape(ori_shape)
    
    
    def transform(self, data_all, n_inputs, n_outputs):
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

    def prep_test_data(self, input_path):
        return
    
    def prep_training_data(self, input_path, stock_index):
        # load numpy file
        npy_file_name = input_path + "/ema{}_beta{}_{}.npy".format(self.ema, self.beta, stock_index)
        input_np_data = np.load(npy_file_name, allow_pickle=True)
        
        # date list
        date_list = []
        for i in range(self.n_training_days):    
            date = input_np_data[i][0][5].date().strftime("%y%m%d")
            date_list.append(date_list)
        
        
        # check if we have days more than training period
        assert(input_np_data.shape[0] >= self.n_training_days)
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
            input_np_data = input_np_data[:,7:-5,:]
            
        shape = input_np_data.shape
        n_training_sequences = self.n_training_days
        if self.split_daily_data == 1:
            assert(shape[1] % 2 == 0)
            input_np_data = input_np_data.reshape((shape[0]*2, 
                                                  int(shape[1]/2), 
                                                  shape[2]))
            # get the first date and last date
            n_training_sequences *= 2
            
        # to scale the data, but not the timestamp and price
        data_train_input, data_train_output = self.transform(input_np_data[:n_training_sequences,:,:-2], len(input_columns), 1)
        return data_train_input, data_train_output, input_np_data[:n_training_sequences,:,-2], input_np_data[:n_training_sequences,:,-1]


# In[8]:


class StrategyDesc:
    def __init__(self, 
                 buy_threshold,
                 sell_threshold,
                 stop_loss,
                 stop_gain,
                 min_hold_steps):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.stop_loss = stop_loss
        self.stop_gain = stop_gain
        self.min_hold_steps = min_hold_steps
        
    def get_parameter_str(self):
        s = "buy_threshold:{} sell_threshold:{} stop_loss:{}             stop_gain:{} min_hold_steps:{}".format(self.buy_threshold,
                                                  self.sell_threshold,
                                                  self.stop_loss,
                                                  self.stop_gain,
                                                  self.min_hold_steps)
        return s
    
    


# In[23]:


import numpy as np
from pathlib import Path
import pandas as pd
import GPy
import GPyOpt

class ValueModel:
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': (10,20,30,40)},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (1,2,5,10)},
          {'name': 'n_repeats', 'type': 'discrete', 'domain': (3,5,10,20,30,40)},
          {'name': 'beta', 'type': 'discrete', 'domain': (99, 98)},
          {'name': 'ema', 'type': 'discrete', 'domain': (1,5,10,20)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
         ]
    
    mixed_domain_test = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': (10,20)},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': (5,10)},
          {'name': 'n_repeats', 'type': 'discrete', 'domain': (3,5)},
          {'name': 'beta', 'type': 'discrete', 'domain': (99, 98)},
          {'name': 'ema', 'type': 'discrete', 'domain': (1,5,10,20)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)}
         ]
    
    
    def __init__(self, stock_name, stock_index, n_training_days):
        self.stock_name = stock_name
        self.stock_index = stock_index
        self.n_training_days = n_training_days
        self.save_path = "model_{}_{}".format(stock_name, n_training_days)
        self.last_training_date = None
        self.model = None
        self.max_profit = -999.0
        return
    
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str
    
    def get_max_steps(self, groups):
        max_steps = 0
        for index, df in groups:
            df_len = len(df)
            if df_len > max_steps:
                max_steps = df_len
        return max_steps

    
    def get_data_prep_desc_filename(self, path):
        return path + '/data_prep_desc.pkl'
    
    def optimize(self, input_csv_path, max_iter=300, is_test=False):
        if is_test == True:
            mixed_domain = self.mixed_domain_test
        else:
            mixed_domain = self.mixed_domain
        
        opt_handler = GPyOpt.methods.BayesianOptimization(f=self.opt_func,  # Objective function       
                                     domain=mixed_domain,          # Box-constraints of the problem
                                     initial_design_numdata = 20,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True)           # True evaluations, no sample noise
        opt_handler.run_optimization(max_iter, eps=0)
    
    def save(self):
        self.model.save(self.save_path, self.last_training_date)
        
    def load(self):
        save_path = self.save_path
        
        # iterate the path, and find out the latest date
        
    
    def opt_func(self, X_list):
        assert(len(X_list)==1)
        answer = np.zeros((X_list.shape[0], 1))
        for i in range(len(X_list)):
            print(self.get_parameter_str(X_list[i]))
            features = X_list[i]
            error, model, value_price = self.get_value_result(features)
            
            strategy_model = StrategyModel()
            strategy_model.optimize(value_price)
            profit, is_hold = strategy_model.get_best_result()
            print("total profit={},  error={}".format(profit, error))
            #self.draw_step_profit_graph(self.step_profit_list, "step_profit_{}".format(answer[i][0]))
            #self.step_profit_list = []
            if profit > self.max_profit:
                strategy_desc = strategy_model.get_strategy_desc()
                
                print("find new opt:{}, {}, {}".format(profit, 
                                                   self.get_parameter_str(X_list[i]),
                                                   strategy_desc.get_parameter_str()))
                self.model = model
                self.save()
                self.max_profit = profit
                
                strategy_model.save(self.save_path)
                
                
                # save the data into file for further analysis
                np_data_to_save = np.concatenate((value_price, is_hold), axis=2)
                print("np_data_to_save")
                print(np_data_to_save.shape)
                np.save(self.save_path + '/best_policy_data.npy', np_data_to_save)
                
                # check the optimized strategy for this model
            answer[i][0] = -profit
        return answer


    
    
    def get_value_result(self, features):
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
        
        assert((self.n_training_days - learning_period) % prediction_period == 0)
        data_manipulator = DataManipulator(beta, ema, 
                                           time_format, 
                                           volume_input, 
                                           use_centralized_bid, 
                                           split_daily_data, 
                                           self.n_training_days)
        npy_path = 'npy_files'
        data_training_input, data_training_output, timestamps, price             = data_manipulator.prep_training_data(npy_path, self.stock_index)
        
        # get the date list.
        date_list = []
        for i in range(len(timestamps)):
            date = timestamps[i][0].strftime("%y%m%d")
            date_list.append(date)
        

        
        # now define the network
        model = StatefulLstmModel(n_neurons, learning_rate, num_layers, rnn_type, n_repeats)
        
        assert(self.n_training_days % prediction_period == 0)
        
        n_training_seq = self.n_training_days
        n_learning_seq = learning_period
        n_prediction_seq = prediction_period
        if split_daily_data == 1:
            n_training_seq *= 2
            n_learning_seq *= 2
            n_prediction_seq *= 2
            
        self.last_training_date = date_list[-1]
        daily_errors = []
        all_outputs = []
        print("start training: training_seq:{}, learning_seq:{}, prediction_seq:{} last_training_date:{}".format(n_training_seq, 
                                                                                           n_learning_seq, 
                                                                                           n_prediction_seq,
                                                                                           self.last_training_date))
        for i in range(0, n_training_seq-n_learning_seq+1, n_prediction_seq):
            learning_end = i + n_learning_seq
            print("start training from seq:{}({}) - seq:{}({})".format(i, date_list[i], learning_end-1, date_list[learning_end-1]))
            model.fit(data_training_input[i:learning_end], data_training_output[:learning_end], n_prediction_seq)
            prediction_end = learning_end + n_prediction_seq
            if prediction_end > n_training_seq:
                break
            
            print("start predicting from seq:{}({}) - seq:{}({})".format(learning_end, date_list[learning_end], 
                                                                       prediction_end-1, date_list[prediction_end-1]))
            
            outputs = model.predict_and_verify(data_training_input[learning_end:prediction_end], 
                                     data_training_output[learning_end:prediction_end])
            print("output.shape")
            print(outputs.shape)
            all_outputs.append(outputs)
            # calculate the error for every day
            y = data_training_output[learning_end:prediction_end]
            # error is a 1-D array for the every day error
            error = np.mean(np.square(outputs-y), axis=(1,2))
        
            daily_errors += error.tolist()
            
        np_all_outputs = np.array(all_outputs)
        print("np_all_outputs.shape")
        print(np_all_outputs.shape)
        shape = np_all_outputs.shape
        
        n_predicted_days = self.n_training_days - learning_period
        if split_daily_data == 1:
            steps_per_day = data_training_input.shape[1] * 2
        else:
            steps_per_day = data_training_input.shape[1]
        
        
        np_all_outputs = np_all_outputs.reshape((n_predicted_days, steps_per_day,1))
        np_all_outputs = data_manipulator.inverse_transform_output(np_all_outputs)
        
        print("np_all_outputs.shape")
        print(np_all_outputs.shape)
        shape = timestamps.shape
        timestamps = timestamps.reshape((self.n_training_days, steps_per_day, 1))
        price = price.reshape((self.n_training_days, steps_per_day, 1))
        
        print("timestamps.shape")
        print(timestamps.shape)
        value_with_timestamp_price = np.concatenate((timestamps[learning_period:],
                                               np_all_outputs,
                                               price[learning_period:]), axis=2)
        print("value_with_timestamp_price")
        print(value_with_timestamp_price.shape)
        ema = get_ema(daily_errors, int(len(daily_errors)/2))
        print("test finished, the ema of testing error:{}".format(ema))
        
        return ema, model, value_with_timestamp_price
    


# In[26]:


class StrategyModel:
    mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0.0, 0.005)},
                 {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.005, 0.0)},
                 {'name': 'stop_loss', 'type': 'continuous', 'domain': (-0.01,-0.003)},
                 {'name': 'stop_gain', 'type': 'continuous', 'domain': (0.002, 0.01)},
                 {'name': 'min_hold_steps', 'type': 'discrete', 'domain': range(10,100)},
         ]
    def __init__(self):
        self.max_profit = -999.0
        self.strategy_desc = None
        return

    def optimize(self, input_data):
        self.input_data = input_data
        
        myBopt = GPyOpt.methods.BayesianOptimization(self.get_profit,  # Objective function       
                                             domain=self.mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = 30,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True,
                                             maximize = True)           # True evaluations, no sample noise

        myBopt.run_optimization(300,eps=0)
        return 0
        
    # the input data is in shape (days, steps, [timestamp, value, price])
    def get_profit(self, X_list):
        assert(len(X_list)==1)
        buy_threshold = X_list[0][0]
        sell_threshold = X_list[0][1]
        stop_loss = X_list[0][2]
        stop_gain = X_list[0][3]
        min_hold_steps = int(X_list[0][4])
        tot_profit = 1
        tot_stock_profit = 1
        buy_step = None
        max_trades = 3
        cost = 0.00015
        n_tot_trades = 0
        
        # to prepare the result data
        shape = self.input_data.shape
        is_hold = np.zeros((shape[0], shape[1], 1))
        daily_profit_list = []
        for day_idx in range(len(self.input_data)):
            #print("starting day {}".format(day_idx))
            n_trades = 0
            daily_profit = 1
            state = 0
            daily_data = self.input_data[day_idx]
            for step in range(len(daily_data)):
                value = daily_data[step][1]
                price = daily_data[step][2]
                time = daily_data[step][0]
                
                if state == 0 and time.time().hour >= 9 and                     n_trades < max_trades and step < len(daily_data)-min_hold_steps:
                    if value > buy_threshold:
                        buy_price = price
                        buy_step = step
                        #print("buy at step {} price:{}".format(step, price))
                        state = 1

                elif state == 1:
                    profit = (price - buy_price)/buy_price
                    if (value < sell_threshold and 
                        step - buy_step > min_hold_steps) or step == len(daily_data)-1 or \
                        profit < stop_loss or \
                        profit > stop_gain:
                        
                        if profit < stop_loss:
                            n_trades = max_trades
                        
                        #print("sell at step {} price:{}".format(step, price))
                        profit -= cost
                        tot_profit *= (1+profit)
                        daily_profit *= (1 + profit)
                        state = 0
                        n_trades += 1
                
                if state == 1:
                    is_hold[day_idx][step] = 1
                else:
                    is_hold[day_idx][step] = 0
            daily_profit_list.append(daily_profit - 1)
            n_tot_trades += n_trades
            last = daily_data[-1][2]
            open = daily_data[0][2]
            stock_profit = (last - open) / open
            tot_stock_profit *= (1+stock_profit)
        
            #print("finishing day {}, daily_profit:{}".format(day_idx, daily_profit))
        #print("{}, n_tot_trades:{} profit:{}".format(X_list, n_tot_trades, tot_profit))
        window = int(len(daily_profit_list)/2)
        profit_ema = get_ema(daily_profit_list, window)
        if profit_ema > self.max_profit:
            print("find best profit ema:{} tot_profit:{} in days:{}".format(profit_ema, 
                                                                            tot_profit,
                                                                            window*2))
            
            self.max_profit = profit_ema
            self.is_hold = is_hold
            self.strategy_desc = StrategyDesc(buy_threshold,
                                             sell_threshold,
                                             stop_loss,
                                             stop_gain,
                                             min_hold_steps)
        
        return np.array(profit_ema).reshape((1,1))
    
    def get_best_result(self):
        return self.max_profit, self.is_hold
    
    
    def get_strategy_desc(self):
        return self.strategy_desc
    
    def get_save_filename(self, path):
        if path[-1] != '/':
            path += '/'
        
        return path + 'strategy_desc.pkl'
        
    
    def save(self, save_path):
        assert(self.strategy_desc != None)
        with open(self.get_save_filename(save_path), 'wb') as f:
            pickle.dump(self.strategy_desc, f, pickle.HIGHEST_PROTOCOL)


# In[ ]:


value_model = ValueModel('Nordea', 5, 60)
value_model.optimize('.', is_test=False)


# In[ ]:




