import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager

if len(sys.argv) < 5:
	print("usage: python3 search-worms.py stock_name, stock_index, start_day_index end_day_index [is_test]")
	sys.exit()

stock_name = sys.argv[1]
stock_index = int(sys.argv[2])
start_day_index = int(sys.argv[3])
end_day_index = int(sys.argv[4])
is_test = False
if len(sys.argv) == 6:
	is_test = bool(sys.argv[5])

stock_worm_manager = StockWormManager(stock_name, stock_index, '../../models', '../../preprocessed-data')
stock_worm_manager.search_worms(0, 60, is_test=is_test)
