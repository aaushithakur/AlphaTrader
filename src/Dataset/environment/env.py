import sys
import os
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
import Shared.configs.config_main as main_config
import Shared.configs.RL.PPO.PPO_config as ppo_config
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")
from Dataset.yfinance.data_process import get_feature_df_RL, fetch_stock_data_RL
import pandas as pd
import talib as ta
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA

def add_TALib_indicator(df, attribute, indicator_func, *args):
    '''
    Adds a column to a dataframe:
        column name is the name of the technical indicator as specified by indicator_func
        column content is the function calculated on the attribute column
    Example: add_TALib_indicator(df, 'AdjClose', ta.RSI, 14) creates a new column called RSI with 
             the 14 day RSI of the values of the column 'AdjClose'
    Inputs:
        df - dataframe - needs to be sorted in date ascending order
        attribute - column name to be used in TA-Lib calculation
        indicator_func - name of a TA-Lib function
        *args - optional parameters for indicator_func
        
    Oupputs:
        df - datarame with new column added
        func_name - name of the new colunm
    
    '''
    # get the name of the indicator from TA-Lib
    func_name = attribute + indicator_func.__name__ + str(*args)
    
    # add new column, calculated based on attribute column
    df.loc[:, func_name] = indicator_func(df.loc[:, attribute].values, *args)
    
    return df, func_name

def add_comparison_cols_for_indicator(df, base_col_name, indicator_col_name, delete_indicator_col=True):
    '''
    adds columns that compare indicator_col to base_col: ratio, crossover, above/below
    Inputs:
        df - dataframe
        base_col_name - name of column that the indicator will get compared to
        indicator_col_name - name of column that has indicator values
        delete_base_col - yes/no on if to keep the base col or not
    Output:
        df - modified df with added & removed columns
    '''
   
    # indicator to base column ratio:
    df.loc[:, indicator_col_name + '_to_' + base_col_name + '_ratio'] = df.loc[:, indicator_col_name] / df.loc[:, base_col_name]
    
    # base col above indicator:
    base_above_indicator_col_name = base_col_name + '_above_' + indicator_col_name
    df.loc[:, base_above_indicator_col_name] = df.loc[:, indicator_col_name] < df.loc[:, base_col_name]
    
    # did base cross indicator
    base_crossed_indicator_col_name = base_col_name + '_crossed_' + indicator_col_name
    df.loc[:, base_crossed_indicator_col_name] = df.loc[:, base_above_indicator_col_name] != df.loc[:, base_above_indicator_col_name].shift(1)
    
    if delete_indicator_col:
        df = df.drop(columns=indicator_col_name)
    
    return df

def feat_eng_changes_values_to_change(df, cols_set_vals_to_change, delete_original_cols=True):
    '''
    Instead of the actual values in some columns, we care about the change from one day to the next.
    This function calculates that change for the given columns and then either keeps or drops (default) the origianl columns
    Input:
        df - a dataframe
        cols_set_vals_to_change - names of columns to work on.
        delete_original_cols - keep or delete original columns
    Output:
        df - dataframe with new columns added. the value in row N is now the change from row N-1 to row N (instead of the actual values)
    '''    

    # calculate the change from row N-1 to row N
    df_chg_cols = (df[cols_set_vals_to_change] / df[cols_set_vals_to_change].shift(1) - 1)

    # add suffix to the column names
    df_chg_cols = df_chg_cols.add_suffix('_chg')

    # join the data onto the original data fram
    df = df.join(df_chg_cols)

    if delete_original_cols:
        # drop the original columns
        df = df.drop(columns=cols_set_vals_to_change)
        
    return df

def scale_env(df, ticker, is_transformer = False):
    if not is_transformer:
        df = df.iloc[:-200] 
    scaler = StandardScaler()
    scaled_df = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)

    if is_transformer:
        df.to_csv(f"{base_path}/ML/Transformers/data/test/transformer_data_full_"+ ticker +".csv", index=True)
        df_transformer_train = df.iloc[:-200]
        scaler_transformer_train = StandardScaler()
        scaled_df_train = pd.DataFrame(scaler_transformer_train.fit_transform(df_transformer_train), index=df_transformer_train.index, columns=df_transformer_train.columns)
        scaled_df_train.round(6)
        pickle.dump(scaler_transformer_train, open(f'{base_path}/ML/Transformers/data/scaler/scaler_transformer_data_train_' + ticker + '.pkl','wb'))
        scaled_df_train.to_csv(f"{base_path}/ML/Transformers/data/train/transformer_data_train_"+ ticker +".csv", index=True)
        return
    
    pickle.dump(scaler, open(f'{base_path}/ML/RL/environment/data/scaler/scaler_' + ticker + '.pkl','wb'))
    pca = PCA(n_components=ppo_config.STATE_SIZE) #number of variables
    principalComponents = pca.fit_transform(scaled_df) #apply PCA
    original_index = scaled_df.index
    scaled_df = pca.transform(scaled_df)
    pickle.dump(pca, open(f"{base_path}/ML/RL/environment/data/pca/pca_" + ticker.upper() + ".pkl","wb"))
    scaled_df = pd.DataFrame(data=scaled_df, index=original_index)
    scaled_df = scaled_df.round(6)
    scaled_df.to_csv(f"{base_path}/ML/RL/environment/data/train/"+ ticker +".csv", index=True)

def create_environment(ticker, is_transformer=False):
    rowData = main_config.ROW_DATA
    path = f"{base_path}/Dataset/stocks/" + ticker + ".csv"
    print()
    df = pd.read_csv(path, header=0)
    #df['Date'] = pd.to_datetime(df.Date, format='%d-%m-%Y')
    df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df = df.set_index('Date')
    df = df.iloc[rowData:]
    
    df_X_base_data, indicator_name = add_TALib_indicator(df, 'AdjClose', ta.RSI, 14)
    # add threshold columns for above 80 and below 20
    df_X_base_data.loc[:, 'RSI_above_80'] = df_X_base_data.loc[:, indicator_name] > 80
    df_X_base_data.loc[:, 'RSI_below_20'] = df_X_base_data.loc[:, indicator_name] < 20
    # normalize to values between 0 and 1
    df_X_base_data.loc[:, indicator_name] = df_X_base_data.loc[:, indicator_name] / 100
    
    # SMA - Simple Moving Average - 2 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 2)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)

    # SMA - Simple Moving Average - 5 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 5)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)

    # SMA - Simple Moving Average - 10 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 10)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)

    # SMA - Simple Moving Average - 20 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 20)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)

    # SMA - Simple Moving Average - 30 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 30)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)

    # SMA - Simple Moving Average - 50 day window
    df_X_base_data, indicator_name = add_TALib_indicator(df_X_base_data, 'AdjClose', ta.SMA, 50)
    df_X_base_data = add_comparison_cols_for_indicator(df_X_base_data, 'AdjClose', indicator_name, delete_indicator_col=False)
    
    cols_set_vals_to_change = ['AdjVolume', 'AdjOpen', 'AdjLow', 'AdjHigh', 'AdjClose', 'AdjCloseSMA5', 'AdjCloseSMA10', 'AdjCloseSMA20', 'AdjCloseSMA30', 'AdjCloseSMA50']
    df_X_base_data = feat_eng_changes_values_to_change(df_X_base_data, cols_set_vals_to_change, delete_original_cols=False)

    df_X_base_data = df_X_base_data.loc[df_X_base_data.notnull().all(axis=1), :]
    df_X_base_data = df_X_base_data.round(6)
    # return df_X_base_data
    df_X_base_data.to_csv(f"{base_path}/ML/RL/environment/data/test/"+ ticker +".csv", index=True)
    scale_env(df_X_base_data, ticker, is_transformer=is_transformer)

def get_environment_train(ticker):
    create_environment(ticker)
    df = pd.read_csv(f"{base_path}/ML/RL/environment/data/train/" + ticker + ".csv", header=0)
    return df

def get_environment(ticker):
    create_environment(ticker)
    df = pd.read_csv(f"{base_path}/ML/RL/environment/data/test/" + ticker + ".csv", header=0)
    return df

def get_ohlcv(ticker, start_date, end_date):
    ohlcv = fetch_stock_data_RL(ticker, start_date, end_date)
    return ohlcv

def get_features_RL(ticker):
    feature_df = get_feature_df_RL(ticker, 10, main_config.STOCK_DATA_START, main_config.STOCK_DATA_END)
    feature_df.index = feature_df.index.date
    return feature_df

def get_transformer_train(ticker, is_transformer=True):
    create_environment(ticker, is_transformer=is_transformer)
    df = pd.read_csv(f"{base_path}/ML/Transformers/data/train/transformer_data_train_"+ ticker +".csv", header=0)
    df.set_index('Date', inplace=True)
    return df

def get_transformer_test(ticker, is_transformer=True):
    create_environment(ticker, is_transformer=is_transformer)
    df = pd.read_csv(f"{base_path}/ML/Transformers/data/test/transformer_data_full_"+ ticker +".csv", header=0)
    df.set_index('Date', inplace=True)
    return df
