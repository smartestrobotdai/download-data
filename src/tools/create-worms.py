import sys
import os.path
sys.path.append("../")
from stockwormmanager import StockWormManager

if len(sys.argv) != 3:
	print("usage: python3 create-worms.py start_day_index end_day_index")
	sys.exit()

start_day_index = int(sys.argv[1])
end_day_index = int(sys.argv[2])



stock_worm_manager = StockWormManager('Nordea', 5, '../../models', '../../preprocessed-data')
swarm_path = stock_worm_manager.get_swarm_path(start_day_index, end_day_index)
if not os.path.isdir(swarm_path):
	print("{} does not exist, aborting...".format(swarm_path))
	sys.exit()

stock_worm_manager.create_worms_from_cache(n_number=2, 
            						start_day_index=start_day_index, 
            						end_day_index=end_day_index)





