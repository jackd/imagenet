from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np


def vis_image(image, bboxes=None, ax=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    if bboxes is not None and len(bboxes.shape) == 1:
        bboxes = np.reshape(bboxes, (-1, 4))
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    if ax is None:
        ax = plt.gca()
    ax.imshow(image)
    if bboxes is not None:
        for bb in bboxes:
            ymin, xmin, ymax, xmax = bb
            rect = patches.Rectangle(
                (xmin, ymin), xmax-xmin, ymax-ymin, facecolor='none',
                linewidth=1, edgecolor='r')
            ax.add_patch(rect)
