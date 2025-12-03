import matplotlib.pyplot as plt

import numpy as np
from PIL import Image

cmap = plt.get_cmap("tab20")
colors = [tuple(int(255 * c) for c in cmap(i)[:3]) for i in range(20)]

bar = []
for i in range(20):
    patch = np.zeros((32, 32, 3))
    patch[:, :, :] = np.array(colors[i])
    bar.append(patch)

bar = np.concatenate(bar, axis=1).astype(np.uint8)
Image.fromarray(bar).save("plt_cmap20_bar.png")


