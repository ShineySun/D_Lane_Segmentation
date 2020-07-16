import os
from pathlib import Path

root_dir = Path(os.getcwd()) / '..'
print(root_dir)

data_root = root_dir

ext = '.pkl'
input_size = 320
data_folder = data_root / 'processed' / 'processed'

cacheFile = data_root / 'train_cache.ptar'
