# DEFINE MODEL
# IS_SIMPLE_NN = False
# MAX_EPOCHS = 100
IS_SIMPLE_NN = False
MAX_EPOCHS = 300

# Define hyperparameters
EMBED_SIZE = 42
HEADS = 6
NUM_LAYERS = 4
MAX_LEN = 5  # considering 15 days + 1 learnable embedding
OUTPUT_SIZE = 1  # Predicting next 3 days stock closing price
EPOCHS = MAX_EPOCHS+5

# BATCH CONFIG
BATCH_SIZE_TRAIN = 40
TEST_INTERVAL = 5
TRAIN_PROFIT_LOOKAHEAD_DAYS = 1
# TRAIN_PROFIT_PERCENT = 0.3 # 0.5% while creating dataset
TRAIN_PROFIT_PERCENT = -0.65 # 0.5% while creating dataset IGM
# TRAIN_PROFIT_PERCENT = -0.5 # 0.5% while creating dataset SPY
FP_THRESH = 0.5 # theshold for True or False

# SCORE
F1_SCORE_THRHESH = 0.5
PRECISION_THRESHOLD = 0.67
RECALL_THRESHOLD = 0.4

TRAIN_ACCURACY = 0.8


# -FN: 24