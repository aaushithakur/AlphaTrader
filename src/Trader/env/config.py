import os, sys
base_path = os.getcwd() + '/src'
sys.path.append(base_path)
from dotenv import load_dotenv

ENV = os.environ.get('ENV')

if (ENV == 'prod'):
  load_dotenv('src/Trader/env/.env.prod')
else:
  load_dotenv('src/Trader/env/.env.dev')
  
def get_env(key):
  return os.environ.get(key)