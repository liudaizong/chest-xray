import cv2
import pandas as pd
import numpy as np

heatmap_data = pd.read_csv('./map_data.csv')
data1 = heatmap_data.iloc[0][2:].as_matrix()
print data1.shape
print heatmap_data.values[0,2:].shape
img_data1 = data1.astype('float').reshape(32,32)
cam = img_data1 - np.min(img_data1)
cam_img = cam / np.max(cam)
cam_img = np.uint8(255*cam_img)

heatmap = cv2.applyColorMap(cv2.resize(cam_img, (1024,1024)), cv2.COLORMAP_JET)
img = cv2.imread('/home/sfchen/chest/images/%s'%(heatmap_data.ix[0][0]))
result = heatmap * 0.3 + img * 0.5
cv2.imwrite('CAM.jpg', result)