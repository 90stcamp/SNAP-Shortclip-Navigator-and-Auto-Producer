import os 

BASE_DIR = os.path.dirname(__file__)
CONFIG_DIR = os.path.join(BASE_DIR, 'config.json')
API_KEY = os.path.join(BASE_DIR, 'key.json')
OUT_DIR = os.path.join(BASE_DIR, 'results')
STATS_DIR = os.path.join(BASE_DIR, 'statistic')

if not os.path.exists(OUT_DIR):
    os.mkdir(OUT_DIR)

if not os.path.exists(STATS_DIR):
    os.mkdir(STATS_DIR)
    
# if not os.path.exists(FIG_DIR):
#     os.mkdir(FIG_DIR)