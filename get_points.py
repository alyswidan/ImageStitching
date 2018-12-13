import matplotlib.pyplot as plt
import numpy as np
def get_points(image):
    plt.imshow(image)
    points = plt.ginput(4)
    plt.show()
