import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=120)
parser.add_argument('--density', type=float, default=0.60)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--every', type=int, default=3)
parser.add_argument('--output', default='figures/поширення.gif')
args = parser.parse_args()
x, y = np.indices((args.n, args.n), dtype=np.uint64)
with np.errstate(over='ignore'):
    h = np.uint64(args.seed) ^ ((x + 1) * np.uint64(0x9e3779b97f4a7c15)) ^ ((y + 1) * np.uint64(0xd1b54a32d192ed03))
    h = (h ^ (h >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
    h = (h ^ (h >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    h ^= h >> np.uint64(31)
a = (((h >> np.uint64(11)).astype(np.float64) * 2**-53) < args.density).astype(np.uint8)
a[:, 0] *= 2
colors = np.array([[240, 234, 215], [34, 120, 67], [252, 169, 36], [61, 60, 58]], dtype=np.uint8)
frames = []
step = 0
while True:
    if step % args.every == 0 or not (a == 2).any():
        image = Image.fromarray(colors[a]).resize((600, 600), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 135, 26), fill=(255, 255, 255))
        draw.text((8, 6), f'Крок: {step}', fill=(30, 30, 30))
        frames.append(image)
    burning = a == 2
    if not burning.any():
        break
    neighbours = np.zeros_like(burning)
    neighbours[1:] |= burning[:-1]
    neighbours[:-1] |= burning[1:]
    neighbours[:, 1:] |= burning[:, :-1]
    neighbours[:, :-1] |= burning[:, 1:]
    new = a.copy()
    new[burning] = 3
    new[(a == 1) & neighbours] = 2
    a = new
    step += 1
Path(args.output).parent.mkdir(exist_ok=True)
frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=120, loop=0, optimize=True)
print('Кадри:', len(frames), 'кроки:', step)
