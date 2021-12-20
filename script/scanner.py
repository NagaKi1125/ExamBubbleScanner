import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 0}

# tien xu lí -------------------------
# load the image, convert it to grayscale, blur it
# slightly, then find edges
image = cv2.imread('../omr.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray',gray)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
# cv2.imshow('blurred', blurred)
edged = cv2.Canny(gray, 75, 200)
# cv2.imshow('edged', edged)
# show hình ảnh sau khi xử lí
# cv2.imshow("Sau khi xu li", edged)
# cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

docCnt = None
# ensure that at least one contour was found
if len(cnts) > 0:
    # sort the contours according to their size in
    # descending order
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # loop over the sorted contours
    for c in cnts:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if our approximated contour has four points,
        # then we can assume we have found the paper
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to both the
# original image and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
cv2.imshow('paper', paper)
warped = four_point_transform(gray, docCnt.reshape(4, 2))
# cv2.imshow('warped', warped)

# apply Otsu's thresholding method to binarize the warped
# piece of paper
# Thresh_otsu dùng để tính toán ngưỡng (k) hợp lí nhất để phân chia ảnh
thresh = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# cv2.imshow('',thresh)
# find contours in the thresholded image, then initialize
# the list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []
# loop over the contours
for i,c in enumerate(cnts):
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    ar = w / float(h)
    # in order to label the contour as a question, region
    # should be sufficiently wide, sufficiently tall, and
    # have an aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(c)

# cv2.drawContours(warped, questionCnts, -1, (0,120,120), 3)
# cv2.imshow('draw', warped)
# sorting the contours from top to bottom
questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

    # loop over the sorted contours
    for (j, c) in enumerate(cnts):

        # construct a mask that reveals only the current
        # "bubble" for the question
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # apply the mask to the thresholded image, then
        # count the number of non-zero pixels in the
        # bubble area
        mask = cv2.bitwise_and(thresh, thresh, mask=mask)
        total = cv2.countNonZero(mask)
        # if the current total has a larger number of total
        # non-zero pixels, then we are examining the currently
        # bubbled-in answer
        if bubbled is None or total > bubbled[0]:
            bubbled = (total, j)


    # initialize the contour color and the index of the
    # *correct* answer
    color = (0, 0, 255)
    k = ANSWER_KEY[q]
    # check to see if the bubbled answer is correct
    if k == bubbled[1]:
        color = (0, 255, 0)
        correct += 1
    print('ket qua lam: {}'.format(bubbled[1]))

    # draw the outline of the correct answer on the test
    cv2.drawContours(paper, [cnts[k]], -1, color, 3)
    print('cau {}: {} - dap an dung: {}\n------------------'.format(q, bubbled[1], k))

cv2.imshow('result', paper)
cv2.waitKey(0)

print((correct / 5.0) * 100)
