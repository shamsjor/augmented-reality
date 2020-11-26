import numpy as np
import cv2
#from calibration_cam import camera_matrix, dist_coef
camera_matrix = np.array(
    [[3.99076128e+03, 0.00000000e+00, 1.81439978e+03], [0.00000000e+00, 4.00109569e+03, 2.15563656e+03],
     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coefs = np.array([2.31801515e-01, -1.88175934e+00, -1.07811839e-02, 1.55749228e-03, 5.06304153e+00])
def draw(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    img = cv2.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -1)
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 3)
        img = cv2.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
        return img
try:
    feature_extractor = cv2.SIFT_create()
except:
    feature_extractor = cv2.ORB()
model = cv2.imread("qr.jpg")
h = model.shape[0]
w = model.shape[1]
rgb_m = cv2.cvtColor(model, cv2.COLOR_BGR2RGB)
gray_m = cv2.cvtColor(rgb_m, cv2.COLOR_RGB2GRAY)
kp_m, desc_m = feature_extractor.detectAndCompute(gray_m, None)
cap = cv2.VideoCapture("qr1.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('project2.avi', cv2.VideoWriter_fourcc('M', 'J',  'P', 'G'), 10, (width, height))

while True:
    ret, frame = cap.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    kp, desc = feature_extractor.detectAndCompute(gray, None)
    test = cv2.drawKeypoints(gray, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_m, desc, k=2)
    good_match = []
    for m in matches:
        if m[0].distance / m[1].distance < 0.5:
            good_match.append(m)
    good_match_arr = np.asarray(good_match)
    im_matches = cv2.drawMatchesKnn(gray_m, kp_m, gray, kp, good_match[0:30], None,
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    good_kp_m = np.array([kp_m[m.queryIdx].pt for m in good_match_arr[:, 0]]).reshape(-1, 1, 2)
    good_kp = np.array([kp[m.trainIdx].pt for m in good_match_arr[:, 0]]).reshape(-1, 1, 2)
    H, masked = cv2.findHomography(good_kp_m, good_kp, cv2.RANSAC, 5.0)
    h = model.shape[0]
    w = model.shape[1]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dsts = cv2.perspectiveTransform(pts, H)
    frame = cv2.polylines(frame, [np.int32(dsts)], True, 255, 3, cv2.LINE_AA)
    dst = cv2.undistort(frame, camera_matrix, dist_coefs)
    objectPoints = 0.1 * np.array(
        [[1, -1, -1], [1, 0, -1], [2, 0, -1], [2, -1, -1], [1, -1, -2], [1, 0, -2], [2, 0, -2], [2, -1, -2]])
    num, Rs, Ts, Ns = cv2.decomposeHomographyMat(H, camera_matrix)
    #for i in range(len(Ts)):
    Rs1 = Rs[1]
    Ts1 = Ts[1]
    imgpts = cv2.projectPoints(objectPoints, Rs1, Ts1, camera_matrix, dist_coefs)[0]
    #print(imgpts)
    drawn_image = draw(dst, imgpts)
    cv2.namedWindow("result", cv2.WINDOW_NORMAL)
    out.write(drawn_image)
    cv2.imshow("result", drawn_image)
    if cv2.waitKey(1) == ord('s'):
        break 
cap.release()
cv2.destroyAllWindows()
