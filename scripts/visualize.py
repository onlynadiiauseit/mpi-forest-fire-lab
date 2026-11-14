import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument('file')
p.add_argument('--n', type=int, required=True)
p.add_argument('--output', default='figures/поширення.png')
a = p.parse_args()
data = np.fromfile(a.file, dtype=np.uint8)
assert data.size == a.n * a.n
matrix = data.reshape((a.n, a.n))
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['#eee7d5', '#287d47', '#f8a600', '#323232'])
plt.figure(figsize=(7, 7))
plt.imshow(matrix, cmap=cmap, vmin=0, vmax=3, interpolation='nearest')
plt.axis('off')
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(a.output, dpi=180, bbox_inches='tight')
