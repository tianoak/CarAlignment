import matplotlib.pyplot as plt
import numpy as np
import caffe

image = np.array(caffe.io.load_image('/home/htian/caffe-vehicleKP/mine/mine.png', color=True)).squeeze()
plt.imshow(image)
plt.show()
