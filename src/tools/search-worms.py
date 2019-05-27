import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager

if len(sys.argv) < 3:
	print("usage: python3 search-worms.py start_day_index end_day_index [is_test]")
	sys.exit()

start_day_index = int(sys.argv[1])
end_day_index = int(sys.argv[2])
is_test = False
if len(sys.argv) == 4:
	is_test = bool(sys.argv[3])

stock_worm_manager = StockWormManager('Nordel', 5, '../../models', '../../preprocessed-data')
stock_worm_manager.search_worms(0, 60, is_test=is_test)
