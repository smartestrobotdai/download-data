#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import GPy
import GPyOpt
import copy
import time
import datetime

class Model:
    # Network Parameters
    # n_neurons, learning_rate, num_layers, rnn_type(RNN|BasicLSTM|LSTM|LSTM peelhole)
    # Control Parameters
    # risk_aversion - the margin added to the courtage that leads an buy or sell operation
    # learning_period - how many sequences model should learn before predicting next sequences
    # prediction_period - how many sequences the model should predict
    # max_repeats - how many times in maximum the model should learn
    # min_profit - what is the minimum profit in average during training phase, if the minimum is not reached, the model should not predict
    # gamma - what is the gamma used when preprocessing data
    
    step_profit_list = []
    mixed_domain = [{'name': 'n_neurons', 'type': 'discrete', 'domain': tuple(range(20,160,20))},
          {'name': 'learning_rate', 'type': 'discrete', 'domain': (0.001,0.002,0.003,0.004)},
          {'name': 'num_layers', 'type': 'discrete', 'domain': (1,2,3,4)},
          {'name': 'rnn_type', 'type': 'discrete', 'domain': (0,1,2)},
          {'name': 'learning_period', 'type': 'discrete', 'domain': tuple(range(10,41,10))},
          {'name': 'prediction_period', 'type': 'discrete', 'domain': tuple(range(2,11,2))},
          {'name': 'max_repeats', 'type': 'discrete', 'domain': tuple(range(1,52,10))},
          {'name': 'beta', 'type': 'discrete', 'domain': (99, 98)},
          {'name': 'ema', 'type': 'discrete', 'domain': (10,20)},
          {'name': 'time_format', 'type': 'discrete', 'domain': (0,1,2)}, #1 for stepofday, 2 for stepofweek
          {'name': 'volume_input', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'use_centralized_bid', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'split_daily_data', 'type': 'discrete', 'domain': (0,1)},
          {'name': 'related_stock', 'type': 'discrete', 'domain': (0,1)},
         ]
    def __init__(self, regen):
        if regen == False:
            return
        def column_filter(x):
            if x == 'stepofweek':
                return True
            elif 'diff_ema' in x:
                return True
            elif 'volume' in x:
                return True
            elif 'value_ema' in x:
                return True
            else:
                return False
        for ema in (10, 20):
            for beta in (99, 98):
                filename = "data-prep-ema{}-beta{}.csv".format(ema, beta)
                print("pre-processing {}".format(filename))
                data = pd.read_csv(filename, parse_dates=["timestamp"])
                data['dayofweek'] = data['timestamp'].apply(lambda x: x.weekday())
                groups = data.set_index('timestamp').groupby(lambda x: x.date())
                
                # get maximum steps
                max_steps = 0
                for index, df in groups:
                    df_len = len(df)
                    if df_len > max_steps:
                        max_steps = df_len
                        
                np_data = np.zeros((len(groups), max_steps, 30*3+2))
                filtered_columns = list(filter(column_filter, data.columns))
                i = 0
                for index, df in groups:
                    df['stepofday'] = np.arange(0, max_steps)
                    df['stepofweek'] = df['dayofweek'] * max_steps + df['stepofday']
                    np_data[i] = df[filtered_columns + ['stepofweek','stepofday']].to_numpy()
                    i += 1
                    
                numpy_file_name = "np_ema{}_beta{}.npz".format(ema, beta)
                np.savez_compressed(numpy_file_name, np_data)
                

        return
        
    def get_parameter_str(self, X):
        parameter_str = ""
        for i in range(len(self.mixed_domain)):
            parameter_str += self.mixed_domain[i]["name"]
            parameter_str += ':'
            parameter_str += str(X[i])
            parameter_str += ','
        return parameter_str
    
    def reset_graph(self, seed=42):
        tf.reset_default_graph()
        tf.set_random_seed(seed)
        np.random.seed(seed)
        
    
    def log(self, verbose, msg):
        if verbose:
            print(msg)

    def get_batch(self, seq_index, data_train_input, data_train_output):
        X_batch = data_train_input[seq_index:seq_index+1]
        y_batch = data_train_output[seq_index:seq_index+1]
        return X_batch, y_batch
    
    def transform(self, data_all, n_inputs, n_outputs):
        orig_shape = data_all.shape
        data_train_reshape = data_all.reshape((orig_shape[0] * orig_shape[1], orig_shape[2]))
        
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
    
    def inverse_transform_output(self, scaled_outputs):
        outputs_reshaped = scaled_outputs.reshape((scaled_outputs.shape[1], scaled_outputs.shape[2]))
        #outputs = np.exp(self.scaler_output.inverse_transform(outputs_reshaped)) - 1
        outputs = self.scaler_output.inverse_transform(outputs_reshaped)
        return outputs
    
    def inverse_transform_input(self, scaled_inputs):
        inputs_reshaped = scaled_inputs.reshape((scaled_inputs.shape[1], scaled_inputs.shape[2]))
        #inputs_reshaped[:,4:6] = np.exp(self.scaler_input.inverse_transform(inputs_reshaped)[:,4:6]) - 1
        inputs = self.scaler_input.inverse_transform(inputs_reshaped)
        # TODO: the volume and hold should be transformed back.
        return inputs
        
        
    def get_answer(self, features):
        n_neurons = int(features[0])
        learning_rate = features[1]
        num_layers = int(features[2])
        rnn_type = int(features[3])
        learning_period = int(features[4])
        prediction_period = int(features[5])
        max_repeats = int(features[6])
        beta = int(features[7])
        ema = int(features[8])
        time_input = int(features[9])
        volume_input = int(features[10])
        use_centralized_bid = int(features[11])
        split_daily_data = int(features[12])
        related_stock = int(features[13])
        # load data
        file_name = "np_ema{}_beta{}.npz".format(ema, beta)
        data_all = np.load(file_name)['arr_0']
        # pick the data for stock_id
        
        # for the stock 20, the max related stock is 21 (0.94),
        # the medium stock is 18 (0.29), the min related stock is 5 (0.05)
        stock_index = [28]
        
        if related_stock == 1:
            stock_index += [5]
        
        # we must convert the array to 2D
        if use_centralized_bid == 0:
            # remove all the rows for centralized bid. it should be from 9.01 to 17.24, which is 516-12=504 steps
            data_all = data_all[:,7:-5,:]
        
        
        orig_shape = data_all.shape
        print("original shape: ")
        print(orig_shape)

        # make it simple, the step must be even number.
        assert(orig_shape[1] % 2 == 0)
        reshaped_data = data_all.reshape((orig_shape[0] * orig_shape[1], 
                                                          orig_shape[2]))
        
        # the mandatory is the diff.
        input_column_list = [30+i for i in stock_index]
        volume_input_list = stock_index
        if time_input == 1:
            input_column_list = [-1] + input_column_list
        elif time_input == 2:
            input_column_list = [-2] + input_column_list
        if volume_input != 0:
            input_column_list = input_column_list + volume_input_list
            
        output_column_list = [60+i for i in stock_index]
        
        reshaped_data_filtered = reshaped_data[:, input_column_list + output_column_list]
        # for volume we use log.
        if volume_input != 0:
            # the last column is the volume
            last_input_index = len(volume_input_list)
            # we must add 1 to the volume value otherwise log(0) is meaningless.
            reshaped_data_filtered[:, -last_input_index:] = np.log(reshaped_data_filtered[:, -last_input_index:]+1)
        
        n_inputs = len(input_column_list)
        n_outputs = len(output_column_list)
        if split_daily_data == 0:
            data_filtered = reshaped_data[:, input_column_list + output_column_list].reshape((orig_shape[0], 
                                                                                          orig_shape[1], 
                                                                                          n_inputs+n_outputs))

        else:
            # we split day data into 2 parts.
            data_filtered = reshaped_data[:, input_column_list + output_column_list].reshape((orig_shape[0]*2, 
                                                                                          int(orig_shape[1]/2), 
                                                                                          n_inputs+n_outputs))
            learning_period *= 2
            prediction_period *= 2
        
        
        np.nan_to_num(data_filtered, copy=False)
        batch_size = 1
        data_train_input, data_train_output = self.transform(data_filtered, n_inputs, n_outputs)

        # data_train_input in the shape [seq, steps, features]
        days = data_train_input.shape[0]

        max_steps = data_train_input.shape[1]
        print("days={}, max_steps={}".format(days, max_steps))
        self.reset_graph()
        
        X = tf.placeholder(tf.float32, [None, max_steps, n_inputs])
        y = tf.placeholder(tf.float32, [None, max_steps, n_outputs])
        
        layers = None
        if rnn_type == 0:
            layers = [tf.nn.rnn_cell.BasicLSTMCell(n_neurons) 
              for _ in range(num_layers)]
        elif rnn_type == 1:
            layers = [tf.nn.rnn_cell.LSTMCell(n_neurons, use_peepholes=False) 
              for _ in range(num_layers)]
        elif rnn_type == 2:
            layers = [tf.nn.rnn_cell.LSTMCell(n_neurons, use_peepholes=True) 
              for _ in range(num_layers)]
        else:
            print("WRONG")
        cell = tf.nn.rnn_cell.MultiRNNCell(layers)
        
        # For each layer, get the initial state. states will be a tuple of LSTMStateTuples.
        init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, n_neurons])
        state_per_layer_list = tf.unstack(init_state, axis=0)
        rnn_tuple_state = tuple(
            [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
             for idx in range(num_layers)]
        )

        rnn_outputs, new_states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32, 
                                                    initial_state=rnn_tuple_state)
        
        stacked_rnn_outputs = tf.reshape(rnn_outputs, [-1, n_neurons])
        stacked_outputs = tf.layers.dense(stacked_rnn_outputs, n_outputs)
        outputs = tf.reshape(stacked_outputs, [-1, max_steps, n_outputs])
        
        
        loss = tf.reduce_mean(tf.square(outputs - y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op = optimizer.minimize(loss)

        init = tf.global_variables_initializer()

        # now run the model to get answer:
        rnn_states_before_training = np.zeros((num_layers, 2, batch_size, n_neurons))
        graph_data = []
        my_loss_test_list = []
        my_test_results_list = []
        my_test_answers_list = []
        with tf.Session() as sess:
            init.run()
            for learn_end_seq in range(learning_period, 
                                       days - prediction_period, 
                                       prediction_period):
                learning_start_seq = learn_end_seq - learning_period
                tmp_states = np.zeros((num_layers, 2, batch_size, n_neurons))
                for repeat in range(max_repeats):
                    rnn_states = copy.deepcopy(rnn_states_before_training)
                    my_loss_train_list = []
                    train_asset = 1
                    for seq in range(learning_start_seq, learn_end_seq):
                        X_batch, y_batch = self.get_batch(seq, data_train_input, data_train_output)
                      
                        feed_dict = {
                            X: X_batch,
                            y: y_batch,
                            init_state: rnn_states_before_training
                        }
                        
                        my_op, my_new_states, my_loss_train, my_outputs = sess.run([training_op, new_states, loss, outputs], feed_dict=feed_dict)
                        
                        my_loss_train_list.append(my_loss_train)
                        rnn_states = my_new_states
                        if seq - learning_start_seq == prediction_period:
                            # next training loop starts from here
                            tmp_states = copy.deepcopy(rnn_states)
                    my_loss_train_avg = sum(my_loss_train_list) / len(my_loss_train_list)
                    
                    print("{} sequence:{} - {} repeat={} training finished, training MSE={}".format(
                        datetime.datetime.now().time(),
                        learning_start_seq, learn_end_seq, 
                        repeat, my_loss_train_avg))
                # backup the states after training.
                rnn_states_before_training = copy.deepcopy(tmp_states)
                
                
                for seq in range(learn_end_seq, learn_end_seq + prediction_period):
                    X_test, y_test = self.get_batch(seq, data_train_input, data_train_output)
                    feed_dict = {
                        X: X_test,
                        y: y_test,
                        init_state: rnn_states,
                    }
            
                    my_new_states, my_loss_test, my_outputs = sess.run([new_states, loss, outputs], feed_dict=feed_dict)
                    my_loss_test_list.append(my_loss_test)
                    real_outputs = self.inverse_transform_output(my_outputs)
                    real_test = self.inverse_transform_output(y_test)
                    output_and_answer = np.hstack((real_outputs.reshape((max_steps, n_outputs)), 
                                                   real_test.reshape((max_steps, n_outputs))))
                    my_test_results_list.append(output_and_answer)
                    print("sequence:{} test finished, testing MSE={}".format(seq, my_loss_test))
                    rnn_states = my_new_states
            my_loss_test_avg = sum(my_loss_test_list)/len(my_loss_test_list)
            
            return my_loss_test_avg, np.array(my_test_results_list)
                    
    def opt_wrapper(self, X_list):
        answer = np.zeros((X_list.shape[0], 1))
        for i in range(len(X_list)):
            print(self.get_parameter_str(X_list[i]))
            features = X_list[i]
            answer[i][0], results_list = self.get_answer(features)
            #self.draw_step_profit_graph(self.step_profit_list, "step_profit_{}".format(answer[i][0]))
            #self.step_profit_list = []
            if answer[i][0] < self.min_answer:
                print("find new opt:{}, {}".format(answer[i][0], self.get_parameter_str(X_list[i])))
                self.min_answer = answer[i][0]
            else:
                print("find result:{}, {}".format(answer[i][0], self.get_parameter_str(X_list[i])))
        return answer
                
        
    def optimize(self, max_iter=300):
        self.min_answer = 999
        myBopt = GPyOpt.methods.BayesianOptimization(f=self.opt_wrapper,  # Objective function       
                                             domain=self.mixed_domain,          # Box-constraints of the problem
                                             initial_design_numdata = 20,   # Number data initial design
                                             acquisition_type='EI',        # Expected Improvement
                                             exact_feval = True)           # True evaluations, no sample noise
        
        myBopt.run_optimization(max_iter,eps=0)
    
    
    # no optimize, we have already knew the answer. run it and save the results into file.
    def run(self, n_neurons, learning_rate, 
            num_layers, rnn_type, 
            learning_period, prediction_period, 
            max_repeats, beta, ema, time_input, volume_input,
            use_centralized_bid, split_daily_data, related_stock):
        features = [n_neurons, learning_rate, 
            num_layers, rnn_type, 
            learning_period, prediction_period, 
            max_repeats, beta, ema, time_input, volume_input,
            use_centralized_bid, split_daily_data, related_stock]
        
        answer, my_test_result_list = self.get_answer(features)
        print("Finished, result:{}".format(answer))
        return my_test_result_list


# In[3]:


model = Model(True)


# In[3]:



model.optimize()


# In[8]:


#find result:0.00021723691439287255, n_neurons:120.0,learning_rate:0.001,num_layers:2.0,rnn_type:1.0,learning_period:30.0,prediction_period:4.0,max_repeats:51.0,beta:99.0,ema:10.0,time_format:1.0,
#volume_input:0.0,use_centralized_bid:1.0,split_daily_data:1.0,related_stock:1.0

my_test_result_list = model.run(n_neurons = 120, 
                                learning_rate=0.001, 
                                num_layers=2, 
                                rnn_type=1,
                                learning_period=30, 
                                prediction_period=4, 
                                max_repeats=51, 
                                beta=99, 
                                ema=10,
                                time_input=1,
                                volume_input=0,
                                use_centralized_bid=1,
                                split_daily_data=1,
                                related_stock=1)


# In[9]:


np.save('results.np', my_test_result_list)


# In[31]:


filename = "data-prep-ema{}-beta{}.csv".format(10,99)
data = pd.read_csv(filename, parse_dates=["timestamp"])


# In[32]:


data


# In[33]:


ori_shape = my_test_result_list.shape

df = pd.DataFrame(my_test_result_list.reshape(ori_shape[0]*ori_shape[1], ori_shape[2]))


# In[34]:


df


# In[35]:


len(my_test_result_list)


# In[36]:


df.corr()


# In[37]:


groups = data.set_index('timestamp').groupby(lambda x: x.date())


# In[38]:


data


# In[40]:


# start day is the learning period.
ori_shape = my_test_result_list.shape
my_test_result_list = my_test_result_list.reshape(int(ori_shape[0]/2), ori_shape[1]*2, ori_shape[2])
start_day = 30
i = 0
my_df_list = []
for index, df in groups:
    print("handling day: {}".format(i))
    if i >= start_day and i-start_day < len(my_test_result_list):
        print(i)
        my_df = df[['last_20','value_ema_10_beta_99_20','last_5','value_ema_10_beta_99_5']]
        print(len(my_test_result_list[i-start_day,:,0]))
        print(len(my_df))
        my_df['test_y_20'] = my_test_result_list[i-start_day,:,0]
        my_df['test_y_5'] = my_test_result_list[i - start_day,:,1] 
        my_df['valid_y_20'] = my_test_result_list[i - start_day,:,2]
        my_df['valid_y_5'] = my_test_result_list[i - start_day,:,3]
        my_df_list.append(my_df)
        
    i+=1
   


# In[41]:


my_df_list[-1]


# In[43]:


buy_threshold = 9.17555532e-05
sell_threshold = -9.62431518e-04
min_hold_steps = 61
tot_profit = 1
tot_stock_profit = 1
result_list = []
for day_idx in range(len(my_df_list)):
    print("starting day {}".format(day_idx))
    trades = 0
    daily_profit = 1
    state = 0
    df = my_df_list[day_idx]
    for step in range(len(df)):
        if df.iloc[step]['test_y_20'] > buy_threshold and state == 0:
            price = df.iloc[step]['last_20']
            print("buy at step {} price:{}".format(step, price))
            state = 1
            last_step = step
        elif state==1 and ((df.iloc[step]['test_y_20'] < buy_threshold and step - last_step > min_hold_steps) or step == len(df) - 1):
            last_price = price
            price = df.iloc[step]['last_20']
            print("sell at step {} price:{}".format(step, price))
            profit = (price - last_price)/last_price
            tot_profit *= (1+profit)
            daily_profit *= (1 + profit)
            state = 0
            trades += 1
    last = df.iloc[len(df)-1].last_20
    open = df.iloc[0].last_20
    stock_profit = (last - open) / open
    tot_stock_profit *= (1+stock_profit)
    
    result_list.append([day_idx, trades, tot_stock_profit, tot_profit])
    print("finishing day {}, daily_profit:{}".format(day_idx, daily_profit))
print(tot_profit)


# In[44]:


result_df = pd.DataFrame(result_list, columns=['day','trades','stock','profit'])
result_df


# In[45]:


result_df[['day','stock','profit']].plot(x='day')


# In[86]:


result_df[['day','trades']].plot(x='day')


# In[53]:


mixed_domain = [{'name': 'buy_threshold', 'type': 'continuous', 'domain': (0.0, 0.001)},
                {'name': 'sell_threshold', 'type': 'continuous', 'domain': (-0.001, 0.0)},
                {'name': 'min_hold_steps', 'type': 'discrete', 'domain': range(10,100)},
        ]
   

   
def opt_wrapper(X_list):
   
   print(X_list)
   buy_threshold = X_list[0][0]
   sell_threshold = X_list[0][1]
   min_hold_steps = int(X_list[0][2])
   tot_profit = 1
   tot_stock_profit = 1
   last_step = None
   max_trades = 2
   for day_idx in range(len(my_df_list)):
       #print("starting day {}".format(day_idx))
       trades = 0
       daily_profit = 1
       state = 0
       df = my_df_list[day_idx]
       for step in range(len(df)):
           if state == 0 and trades<max_trades:
               if df.iloc[step]['test_y_5'] > buy_threshold:
                   price = df.iloc[step]['last_5']
                   hold = 5
               elif df.iloc[step]['test_y_20'] > buy_threshold:
                   price = df.iloc[step]['last_20']
                   hold = 20
                   #print("buy at step {} price:{}".format(step, price))
               state = 1
               last_step = step
           elif state == 1:
               if (df.iloc[step]['test_y_'+str(hold)] < buy_threshold and 
                   step - last_step > min_hold_steps) or step == len(df)-1:
                   
                   last_price = price
                   price = df.iloc[step]['last_'+str(hold)]
                   #print("sell at step {} price:{}".format(step, price))
                   profit = (price - last_price)/last_price
                   tot_profit *= (1+profit)
                   daily_profit *= (1 + profit)
                   state = 0
                   hold = 0
                   trades += 1
       last = df.iloc[len(df)-1].last_20
       open = df.iloc[0].last_20
       stock_profit = (last - open) / open
       tot_stock_profit *= (1+stock_profit)
       #print("finishing day {}, daily_profit:{}".format(day_idx, daily_profit))
   print("{}, profit:{}".format(X_list, tot_profit))
   return -tot_profit
   


# In[49]:


myBopt = GPyOpt.methods.BayesianOptimization(opt_wrapper,  # Objective function       
                                     domain=mixed_domain,          # Box-constraints of the problem
                                     initial_design_numdata = 20,   # Number data initial design
                                     acquisition_type='EI',        # Expected Improvement
                                     exact_feval = True)           # True evaluations, no sample noise

myBopt.run_optimization(100,eps=0)


# In[82]:


my_df[['test_y','valid_y']].corr()


# In[83]:


my_df['diff']=my_df['test_y']-my_df['valid_y']


# In[84]:


my_df['diff'].plot()


# In[85]:


my_df['diff'].plot.kde()


# In[86]:


my_df['diff'].mean()


# In[87]:


my_df.plot.scatter(x='valid_y', y='test_y',s=1)


# In[88]:


my_df['diff'].max()


# In[89]:


my_df.loc[my_df['valid_y'].argmax()]


# In[90]:


my_df


# In[91]:


buy_threshold = 0.0005
sell_threshold = -0.0005

action_steps = my_df[my_df['test_y']>0.0005].append(my_df[my_df['test_y']<-0.0005]).sort_values('timestamp')


# In[92]:


state = 0
profit = 1
for i in range(len(my_df)):
    row = my_df.iloc[i]
    if row['test_y'] > 0.0001 and state == 0:
        print("buy at {} price:{}".format(row.index, row.last_20))
        state = 1
        price = row.last_20
    elif row['test_y'] < -0.0001 and state == 1:
        state = 0
        ratio = (row.last_20 - price) / price
        print("sell at {} price:{}".format(row.index, row.last_20)) 
        if ratio != 0:
            profit = profit * ratio
print("profit={}".format(profit))


# In[69]:


action_steps


# In[ ]:




