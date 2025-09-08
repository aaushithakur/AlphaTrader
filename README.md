# AlphaTrader

Takes you to the moon!

## Description

This is a research project focused on exploring different machine learning approaches that can be used for doing algorithmic trading. There is a stable production version deployed on the server. New state of art algorithms and techniques are continuously researched and integrated. We create the dataset from scratch for all ML techniques including the environments for Reinforcement Learning. There is a market simulator for backtesting the strategies.

## Getting Started

### Dependencies

* Create conda environment using environment.yml
* Use pip to install requirements.txt
* Override the two dependencies with requirements-override.txt

### Machine Learning Techniques:

1. **Simple Neural Network:** There is a neural network (in src/ML/NN) that predicts if the selected stock will go up by 0.65% the next day.

2. **Reinforcement Learning:** There are two agent options (DQN or PPO), which basically learn the optimal policy for asset allocation percentage in the stock for the next day. Agent sells the current holdings at the end of the day and buys the predicted amount after that for the next day. We use **Transformer** to learn the policy for PPO.

3. There is a **Transformer** implemented as well to predict if the open price on the next day will be 0.65% less than close price today or not.

**Prod Strategy:**
Combining the predictions from all the models gives us the production deployed model. Current production strategy is to give the most weight on RL (PPO) prediction. Take the average of top two actions predicted by PPO agent (which will be buying power for that day). Now if RL agent gives buying power of 0 (as the maximum probability, although there will always be two actions so it wont be zero) but Neural Network/SimpleNN/Transformer predict True, we double the average predicted buying power. If the agent predicts 0 as the highest probability and Transformer/NN/SimpleNN give false, instead of taking average, we just set buying_power to 0.

#### Neural Network

1. First run :
    ```
    python src/Dataset/yfinance/fetch_data.py
    ```
    This will get the ohlcv data of the ticker used in config_main in Shared/configs.
2. Then run the src/ML/NN/create_dataset.ipynb notebook to create dataset.
3. Check the TP thresh, FP thresh in auto_train_metric_k_fold.py and Run:
    ```
    python src/ML/NN/auto_train_metric_k_fold.py
    ```
4. Copy the saved model including scaler and pickle (if needed) from model folder to Shared/models/NN

#### Transformers/SimpelNN 

1. First run :
    ```
    python src/Dataset/yfinance/fetch_data.py
    ```
    This will get the ohlcv data of the ticker used in config_main in Shared/configs.
2. Chech the config and change threshold in src/Shared/configs/Transformer/train_config.py for the stock you are training: -0.65: IGM, -0.5: SPY etc 
3. Run engine.py to create dataset and exit. Then copy the scaler from ```
"src/ML/Transformers/data/scaler/scaler_transformer_data_train_IGM.pkl" to "Shared/models/Transformers/"```
4. Then after setting Shared/configs/Transformers/train_config.py run:
    ```
    python src/ML/transformers/engine.py
    ```
5. This will create your features, scaling etc automatically. Now you should see that training starts. 
6. Copy the saved model including scaler and pickle (if needed) from model folder to Shared/models/Transformers

#### Reinforcement Learning (PPO) -> TRAIN FOR ABOUT (>900) EPISODES for this one

1. First run :
    ```
    python src/Dataset/yfinance/fetch_data.py
    ```
    This will get the ohlcv data of the ticker used in config_main in Shared/configs.
2. Set IS_TRAINING_PPO to True in sim_config.py
3. Check the reward settings in PPO_config.py
5. Then after setting (MAKE SURE THE ACTION SPACE IS CORRECT) Shared/configs/RL/PPO/PPO_config.py run:
    ```
    python src/ML/RL/PPO/train.py
    ```
    This will create your features, scaling etc and training will start.
6. The validation score will be printed. usually you need to go untill the reward becomes saturated about 450-900 episodes. Select the model that has maximum returns in cumulative rewards as well as validation rewards.
7. Copy the saved model including scaler and pickle (if needed) from model folder to Shared/models/RL/PPO


### NOTE 
ALWAYS KEEP  [(-3.46, 10, 13), (-3.45, 172, 175), (-2.72, 143, 146) ... 
less than 3.5 even if the max_portfolio_profit is not highest. 
Set that to be moderate but control losses when training for individual index funds.


### Sample good sim overall
The amount that was not invested on average is 122.16290096989334
prob_good_buy: 0.5979899497487438
max_portfolio_profit: 71.47330052984842
max_portfolio_loss: -3.5481212640850117
best_buy_profit: 8.529889187912971
worst_buy_loss: -2.1750972056071305

max_consecutive_loss: -3.46


max_loss_streak: 5

loss_start_end_days: [(-3.46, 10, 13), (-3.45, 172, 175), (-2.72, 143, 146), (-2.6, 18, 23), (-2.45, 69, 73), (-2.19, 148, 150), (-1.92, 187, 190), (-1.72, 89, 91), (-1.59, 51, 54), (-1.57, 116, 118), (-1.46, 3, 5), (-1.34, 151, 154), (-1.31, 24, 26), (-1.21, 15, 17), (-1.21, 137, 139), (-1.2, 140, 142), (-1.08, 43, 44), (-1.07, 164, 166), (-1.06, 98, 100), (-1.05, 77, 78), (-0.99, 63, 64), (-0.98, 102, 105), (-0.98, 119, 122), (-0.94, 8, 9), (-0.93, 123, 124), (-0.62, 134, 135), (-0.6, 113, 114), (-0.55, 178, 180), (-0.46, 37, 38), (-0.43, 75, 76), (-0.42, 79, 81), (-0.38, 94, 95), (-0.37, 35, 36), (-0.36, 132, 133), (-0.31, 49, 50), (-0.3, 0, 1), (-0.27, 87, 88), (-0.25, 45, 47), (-0.24, 192, 193), (-0.23, 128, 129), (-0.23, 167, 168), (-0.2, 109, 110), (-0.17, 106, 107), (-0.12, 176, 177), (-0.1, 194, 195), (-0.09, 6, 7), (0.0, 157, 158)]

## Backtesting
1. First run :
    ```
    python src/Dataset/yfinance/fetch_data.py
    ```
    This will get the ohlcv data of the ticker used in config_main in Shared/configs.
2. Check the Shared/configs/Backtesting/sim_config.py . The IS_TRAINING is an important flag. Basically, if you want to backtest on prod models, set IS_TRAINING to False, otherwise set it to True.
3. Run: 
    ```
    python src/Backtesting/backtest.py
    ```
4. Do run sim once with profit threshold = 2.1 (better) AND 1.86
5. Do run sim once with profit threshold = 1000 and:
```
    LOSS_THRESHOLD_DICT = {
    'IGM': [-0.75, -1], # [>= 100, <100]
    'SMH': [-0.75, -1], # [>= 100, <100]
    'VUG': [-0.75, -1], # [>= 100, <100]
    'SPY': [-0.5, -0.5] # [>= 100, <100]
    }
```
## Trading
1. For dev, Run: 
    ```
    python src/trader/manage.py run --no-reload
    ```
2. For prod, Run: 
    ```
    # After setting ecosystem.config.js args with:
    # python src/trader/manage.py run --no-reload
    
    pm2 start src/Trader/env/ecosystem.config.js
    ```


## Authors

Contributors names

[@akshat7101999](https://github.com/akshat7101999)
[@aaushithakur](https://github.com/aaushithakur)

## Version History
* 1.0
    * Initial Release

## License

This project is licensed under the Apache License - see the LICENSE.md file for details


