import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager
from util import *

if len(sys.argv) < 5:
	print("usage: python3 search-worms.py stock_name, stock_index, start_day_index end_day_index [search_strategy] [is_test]")
	sys.exit()

stock_name = sys.argv[1]
stock_index = int(sys.argv[2])
start_day_index = int(sys.argv[3])
end_day_index = int(sys.argv[4])

search_strategy = False
if len(sys.argv) >= 6:
	search_strategy = bool(sys.argv[5])


is_test = False
if len(sys.argv) >= 7:
	is_test = bool(sys.argv[6])

stock_data_dir = get_stock_data_dir()
preprocessed_data_dir = get_preprocessed_data_dir()

stock_worm_manager = StockWormManager(stock_name, stock_index, stock_data_dir, preprocessed_data_dir)
swarm_path = stock_worm_manager.get_swarm_path(0, 60)

if not os.path.isdir(swarm_path):
	os.makedirs(swarm_path, exist_ok=True)

stock_worm_manager.search_worms(0, 60, is_test=is_test, search_strategy=True)
