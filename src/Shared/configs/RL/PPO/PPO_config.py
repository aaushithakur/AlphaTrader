import sys, os
base_path = os.getcwd() + '/src'
sys.path.append(base_path)

ROW_DATA = -5000
STATE_SIZE = 32
ACTIONS = [0, 0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.20]
ACTION_SIZE = len(ACTIONS)
POS_REWARD_SCALER = 1 # (ALWAYS POSITIVE)
NEG_REWARD_SCALER = 1 # SPY, SPXX : 1.5, IGM: 1 (ALWAYS POSITIVE)
POS_REWARD = 1 # (ALWAYS POSITIVE)
NEG_REWARD = 1 # SPY, SPXX : 1.5, IGM: 1  # (ALWAYS POSITIVE)
EPS_DECAY = 1 # (ALWAYS POSITIVE)
EPISODES = 200 # > 270 # Transfromer
MEMORY = 60000

#TRANSFORMER
EMBED_SIZE_RLT = STATE_SIZE
HEADS_RLT = 4
NUM_LAYERS_RLT = 2
MAX_LEN_RLT = 1  # considering 15 days + 1 learnable embedding
OUTPUT_SIZE_RLT = 1  # Predicting next 3 days stock closing price