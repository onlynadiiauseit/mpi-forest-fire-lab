import csv
import numpy as np
from scipy.ndimage import label
from pathlib import Path

root = Path(__file__).resolve().parents[1]
n = 128
x, y = np.indices((n, n), dtype=np.uint64)
mask = np.uint64(0xffffffffffffffff)
a = np.uint64(0x9e3779b97f4a7c15)
b = np.uint64(0xd1b54a32d192ed03)

def grid(seed, density):
    with np.errstate(over='ignore'):
        h = np.uint64(seed) ^ ((x + 1) * a) ^ ((y + 1) * b)
        h = ((h ^ (h >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)) & mask
        h = ((h ^ (h >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)) & mask
        h ^= h >> np.uint64(31)
    return ((h >> np.uint64(11)).astype(np.float64) * 2 ** -53) < density

rows = []
for index in range(50, 71):
    d = index / 100
    for seed in range(50):
        trees = grid(seed, d)
        components, _ = label(trees, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8))
        left = np.unique(components[:, 0])
        left = left[left != 0]
        right = np.unique(components[:, -1])
        reached = int(bool(np.intersect1d(left, right).size))
        rows.append({'дослід': 'перколяція', 'версія': 'V0', 'метод': 'зв’язність_4', 'N': n, 'P': 1, 'd': f'{d:.2f}', 'seed': seed, 'повтор': seed + 1, 'час_с': '', 'обмін_с': '', 'обчислення_с': '', 'редукція_с': '', 'кроки': '', 'згоріло': '', 'досягнуто': reached, 'hash': '', 'байти': '', 'статус': 'успішно', 'примітка': 'Незалежна перевірка зв’язності, не вимірювання MPI'})
fields = list(rows[0])
(root / 'measurements').mkdir(exist_ok=True)
with (root / 'measurements' / 'percolation_original.csv').open('w', encoding='utf-8-sig', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('Перевірено', len(rows), 'реалізацій')
for d in [0.5,0.59,0.6,0.7]:
    print(d, sum(int(r['досягнуто']) for r in rows if r['d']==f'{d:.2f}') / 50)
