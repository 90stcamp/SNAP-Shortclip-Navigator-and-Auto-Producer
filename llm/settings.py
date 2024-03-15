import os 

BASE_DIR = os.path.dirname(__file__)

SERVER_DIR = os.path.join(BASE_DIR, 'file/server.json')
CONFIG_DIR = os.path.join(BASE_DIR, 'file/config.json')

API_KEY = os.path.join(BASE_DIR, 'file/key.json')
OUT_DIR = os.path.join(BASE_DIR, 'results')
STATS_DIR = os.path.join(BASE_DIR, 'statistic')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

if not os.path.exists(STATS_DIR):
    os.mkdir(STATS_DIR)
    
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)