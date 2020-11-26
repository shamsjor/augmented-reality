import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
square_size = 2.6
img_mask = './images/*.jpg'
pattern_size = (9, 6)

figsize = (20, 20)
img_names = glob(img_mask)
num_images = len(img_names)

pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size

obj_points = []
img_points = []
h, w = cv2.imread(img_names[0]).shape[:2]
plt.figure(figsize=figsize)

for i, fn in enumerate(img_names):
    print('processing %s... ' % fn)
    imgBGR = cv2.imread(fn)

    if imgBGR is None:
        print("Failed to load", fn)
        continue

    imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

    assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
    found, corners = cv2.findChessboardCorners(img, pattern_size)
   

    if not found:
        print('chessboard not found')
        continue

    if i<12:
        img_w_corners = cv2.drawChessboardCorners(imgRGB, pattern_size, corners, found)
        plt.subplot(4, 3, i+1)
        plt.imshow(img_w_corners)



    print('           %s... OK' % fn)
    img_points.append(corners.reshape(-1, 2))
    obj_points.append(pattern_points)


plt.show()
rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)

print("\nRMS:", rms)
print("camera matrix:\n", camera_matrix)
print("distortion coefficients: ", dist_coefs.ravel())