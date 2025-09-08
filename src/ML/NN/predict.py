import os, sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
sys.path.append(f'{base_path}/ML/NN' )
import pickle
import Shared.configs.config_main as main_config
scaler = {}
mlp = {}
pca = {}

scaler_NN = {}
mlp_NN = {}
pca_NN = {}


for TICKER in main_config.TICKERS:
  scaler[TICKER] = pickle.load(open(f'{base_path}/Shared/models/NN/scaler_{TICKER.lower()}.pkl','rb'))
  # load the model from disk
  mlp[TICKER] = pickle.load(open(f'{base_path}/Shared/models/NN/{TICKER}_model1.sav', 'rb')) #mlp
  pca[TICKER] = pickle.load(open(f"{base_path}/Shared/models/NN/pca_{TICKER.lower()}.pkl",'rb')) #principal component analysis 

  # scaler_NN[TICKER] = pickle.load(open(f'{base_path}/Shared/models/Transformers/scaler_{TICKER.lower()}.pkl','rb'))
  # # load the model from disk
  # mlp_NN[TICKER] = pickle.load(open(f'{base_path}/Shared/models/Transformers/{TICKER}_model1.sav', 'rb')) #mlp
  # pca_NN[TICKER] = pickle.load(open(f"{base_path}/Shared/models/Transformers/pca_{TICKER.lower()}.pkl",'rb')) #principal component analysis 


def should_buy_NN(TICKER, df):
  input_df = scaler[TICKER].transform(df)
  input_df = pca[TICKER].transform(input_df)
  output = mlp[TICKER].predict(input_df)
  prob = mlp[TICKER].predict_proba(input_df)
  for idx in range(len(output)):
    if prob[idx][1] < 0.5:
      output[idx] = False
  # if RANDOM FLAG:
  #   output = [bool(random.getrandbits(1)) for _ in range(len(output))]
  return output

# def should_buy_NN_trans(TICKER, df):
#   input_df = scaler_NN[TICKER].transform(df)
#   input_df = pca_NN[TICKER].transform(input_df)
#   output = mlp_NN[TICKER].predict(input_df)
#   prob = mlp_NN[TICKER].predict_proba(input_df)
#   for idx in range(len(output)):
#     if prob[idx][1] < 0.5:
#       output[idx] = False
#   # if RANDOM FLAG:
#   #   output = [bool(random.getrandbits(1)) for _ in range(len(output))]
#   return output

def pred_uncertainty(TICKER, df):
  input_df = scaler[TICKER].transform(df)
  input_df = pca[TICKER].transform(input_df)
  output = mlp[TICKER].predict(input_df)
  prob = mlp[TICKER].predict_proba(input_df)
  print(prob, output)
  return output
