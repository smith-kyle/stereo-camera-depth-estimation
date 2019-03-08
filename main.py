import numpy as np
import cv2

MIN_MATCH_COUNT = 10

img1 = cv2.imread('images/left.png', 0)          # queryImage
img2 = cv2.imread('images/right.png', 0)  # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 0

flann = cv2.FlannBasedMatcher(
    dict(algorithm=FLANN_INDEX_KDTREE, trees=5), dict(checks=5))

matches = flann.knnMatch(des1, des2, k=2)

good = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.8*n.distance:
        good.append(m)
        pts2.append(kp2[m.trainIdx].pt)
        pts1.append(kp1[m.queryIdx].pt)

pts1 = np.int32(pts1)
pts2 = np.int32(pts2)
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]


def filterWithEpipolarContraints(pts1, pts2, lines):
    newPts1 = []
    newPts2 = []
    for line, pt1, pt2 in zip(lines, pts1, pts2):
        a, b, c = line
        distance = abs(pt1[0] * a + pt1[1] * b + c) / \
            ((a ** 2 + b ** 2) ** (1/2))
        if distance < 0.5:
            newPts1.append(pt1)
            newPts2.append(pt2)

    return np.array(newPts1).T, np.array(newPts2).T


def getError(points, predictedDepths):
    result = []
    with open("./depth.csv") as f:
        lines = f.readlines()
        for line in lines:
            depths = line.split(",")
            depths = [-1.0 if x == "nan" else float(x) for x in depths]
            result.append(depths)

    result = np.array(result).T
    actualDepths = []
    for x, y in points:
        actualDepths.append(result[y, x])
    return np.sqrt(np.mean((predictedDepths-actualDepths)**2))


def predictDepths(pts1, pts2):
    K1 = np.array([
        [1400.06, 0, 984.081],
        [0, 1400.06, 566.927],
        [0, 0, 1]
    ])
    K2 = np.array([
        [1399.22, 0, 910.552],
        [0, 1399.22, 580.636],
        [0, 0, 1]
    ])
    Rt = np.array([
        [0, 0, 0, 119.853],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ])
    P1 = np.matmul(K1, np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]))
    P2 = np.matmul(K2,  Rt)
    points = cv2.triangulatePoints(P1, P2, pts1, pts2)
    # points = np.array(points).T
    result = []
    # for x, y, z, h in points:
    #     result.append(np.array([x, y, z]) / h)

    return np.array(result)


# Find epilines corresponding to points in right image (second image) and
# drawing its lines on left image
lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, F)
lines1 = lines1.reshape(-1, 3)
pts1, pts2 = filterWithEpipolarContraints(pts1, pts2, lines1)
predictedDepths = predictDepths(pts1, pts2)
# print(predictDepths[0])

# img5, img6 = drawlines(img1, img2, lines1, pts1, pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
# lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, F)
# lines2 = lines2.reshape(-1, 3)
# img3, img4 = drawlines(img2, img1, lines2, pts2, pts1)

# plt.subplot(121), plt.imshow(img5)
# plt.subplot(122), plt.imshow(img3)
# plt.show()
