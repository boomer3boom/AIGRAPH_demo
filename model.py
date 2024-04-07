import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.utils import pad_sequences
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K
from keras import losses, optimizers, metrics
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import keras
from torch_geometric_temporal.nn.recurrent import A3TGCN
from torch_geometric_temporal.signal import temporal_signal_split

class CombinedModel:
    def __init__(self, model1, model2):
        self.model1= model1
        self.model2= model2

    def get_test(self, dataset1, dataset2, **kwargs):
        #dataset1
        dataset1['date'] = pd.to_datetime(dataset1['date'], format="%d/%m/%Y")
        dataset1.rename(columns={'index': 'date'}, inplace=True)
        dataset1 = dataset1.drop(['Postcode', 'Type', 'No.Of Sales'], axis =1)
        dataset1.set_index('date', inplace=True)
        dataset1['Id'] = pd.to_numeric(dataset1['Id'])
        dataset1['Id'] -= 1
        dataset1['Price'] = pd.to_numeric(dataset1['Price'])
        dataset1['Price'] = dataset1['Price'].astype('float32')
        for node_id in range(14):
          mask = dataset1['Id'] == node_id
          dataset1.loc[mask, 'Price'] = (dataset1[mask]['Price'] - dataset1[mask]['Price'].mean()) / dataset1[mask]['Price'].std()
        
        truth = []
        for i in range(13):
            node_list = []
            true_values = dataset1[dataset1['Id'] == i]['Price']
            truth.append(true_values.to_list())
        
        
        # Threshold date
        threshold_date1 = pd.to_datetime('2022-09-01')
        threshold_date2 = pd.to_datetime('2022-01-01')
        
        # Filter data using boolean indexing
        X_train = dataset1[dataset1.index < threshold_date1]
        Y_train = X_train['Price']
        
        X_Valid = dataset1[(dataset1.index > threshold_date2) & (dataset1.index < threshold_date1)]
        Y_Valid = X_Valid['Price']
        
        X_Test = dataset1[dataset1.index >= threshold_date1]
        Y_Test = X_Test['Price']
        
        self.X_Test = X_Test
        
        look_back = 6
        
        train_generator = tf.keras.utils.timeseries_dataset_from_array(X_train, Y_train, sequence_length=look_back, batch_size=32)
        valid_generator = tf.keras.utils.timeseries_dataset_from_array(X_Valid, Y_Valid, sequence_length=look_back, batch_size=32)
        test_generator = tf.keras.utils.timeseries_dataset_from_array(X_Test, Y_Test, sequence_length=look_back, batch_size=32)
        
        #dataset2
        train_dataset, test_dataset = temporal_signal_split(dataset2, train_ratio=0.9)
        
        return test_generator, test_dataset, truth

    def predict(self, test_generator, dataset, **kwargs):
        #Model 1
        predictions1 = self.model1.predict(test_generator)
        prediction_df = pd.DataFrame(predictions1, columns=['Predicted Price'])
        #This 4 is in place for look_back
        LookBack_X_Test = self.X_Test[5:]
        prediction_df.index = LookBack_X_Test.index
        prediction_df['Id'] = LookBack_X_Test['Id']
          
        #edit date as required
        threshold_date = pd.to_datetime('2023-03-01')
        prediction_df = prediction_df[prediction_df.index >= threshold_date]
        predictions1 = []
        for i in range(13):
            node_list = []
            prediction_for_id_n = prediction_df[prediction_df['Id'] == i]['Predicted Price']
            predictions1.append(prediction_for_id_n.to_list())
    
        predictions1 = torch.tensor(predictions1)
        predictions1 = [predictions1]
        predictions1[0] = predictions1[0] * 2
    
        #Model 2
        device = torch.device('cpu')
        predictions2 = []
        labels = []
        loss = 0
        step = 0
        horizon = 288
      
        for snapshot in dataset:
            snapshot = snapshot.to(device)
            y_hat = self.model2(snapshot.x, snapshot.edge_index)
            loss = loss + torch.mean((y_hat-snapshot.y)**2)
            labels.append(snapshot.y)
            predictions2.append(y_hat)
            step += 40
            if step > horizon:
              break
    
        predictions2[0] = predictions2[0] * 2
        predictions = 0.5 * predictions1[0] + 0.5 * predictions2[0]
        predictions = [predictions]
        return predictions, labels

