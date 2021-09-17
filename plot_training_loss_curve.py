

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

r1_filename = '/media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_bs=24_lr=0.1_tiny_224_4_/log_2021-09-09-10-36.csv'
r2_filename = '/media/ubuntu/Data/VesselSeg-Pytorch/gd/gd_lineseg_new_Swin_Unet_V2_bs=24_lr=0.2_tiny_224_4_/log_2021-09-09-18-06.csv'


def get_data(filename):
    lines = []
    with open(filename, 'r') as fp:
        lines = [line.strip() for line in fp.readlines()]
    lines = lines[1:]
    print(lines)
    epochs = []
    losses = []
    for line in lines:
        epoch, loss = line.split(',')
        epochs.append(int(epoch))
        losses.append(float(loss))
    return epochs, losses


epochs, losses1 = get_data(r1_filename)
epochs, losses2 = get_data(r2_filename)

fig, ax = plt.subplots(1, figsize=(8,6))

# fig.suptitle("Training loss curve", fontsize=15)

ax.plot(epochs, losses1, color="red", label="Swin-Unet")
ax.plot(epochs, losses2, color="blue", label="Swin-Unet-M")
plt.legend(loc="upper right", title="", frameon=False)

axins = zoomed_inset_axes(ax, 2, loc="center right")  # zoom = 6
axins.plot(epochs,losses1, color="red")
axins.plot(epochs,losses2, color="blue")

# sub region of the original image
x1, x2, y1, y2 = 45, 50, 0.030, 0.035
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
# fix the number of ticks on the inset axes
axins.yaxis.get_major_locator().set_params(nbins=5)
axins.xaxis.get_major_locator().set_params(nbins=5)

# plt.xticks(visible=True)
# plt.yticks(visible=True)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


plt.savefig("/media/ubuntu/Data/VesselSeg-Pytorch/losscurve.png")
plt.show()







