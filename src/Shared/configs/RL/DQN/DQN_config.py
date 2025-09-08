import sys, os
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
import Shared.configs.config_main as main_config

LOOKBACK_INTERVAL = 10
ROW_DATA = -5000
STATE_SIZE = 32
ACTIONS = [0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.20]
ACTION_SIZE = len(ACTIONS)
POS_REWARD_SCALER = 1 
NEG_REWARD_SCALER = 1.5
# modest
POS_REWARD = 4
NEG_REWARD = 1
EPS_DECAY = 1 
EPISODES = 200 #60-100 #125-135  overfit may be afterwards
MEMORY = 60000
