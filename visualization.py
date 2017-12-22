from __future__ import absolute_import, division, print_function
import numpy as np
import matplotlib.pyplot as plt

losses = np.loadtxt('./logs/losses_collection.txt', dtype=np.float32) * 1e6

f, axarr = plt.subplots(2, 2)

axarr[0, 0].plot(losses[:, 0], losses[:, 1])
axarr[0, 0].set_title('BBox regression')
axarr[0, 1].plot(losses[:, 0], losses[:, 2])
axarr[0, 1].set_title('IoU regression')
axarr[1, 0].plot(losses[:, 0], losses[:, 3])
axarr[1, 0].set_title('Classification')
axarr[1, 1].plot(losses[:, 0], losses[:, 4])
axarr[1, 1].set_title('Total loss')

# Fine-tune figure; hide x ticks for top plots and y ticks for right plots
plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

plt.show()
