from PyQt5 import QtWidgets, QtCore, QtGui
from ui import layout
import sys
import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform
from imutils import contours

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def convert_cv_qt(cv_img):
    """Convert from an opencv image to QPixmap"""
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
    p = convert_to_Qt_format.scaled(256, 450, QtCore.Qt.KeepAspectRatio)
    return QtGui.QPixmap.fromImage(p)


def getAnswerChar(key):
    if key == 0:
        return 'A'
    elif key == 1:
        return 'B'
    elif key == 2:
        return 'C'
    elif key == 3:
        return 'D'
    elif key == 4:
        return 'E'


def changeText(char):
    if (char == 'A') or (char == 'a') or (char == '0'):
        return 0
    elif (char == 'B') or (char == 'b') or (char == '1'):
        return 1
    elif (char == 'C') or (char == 'c') or (char == '2'):
        return 2
    elif (char == 'D') or (char == 'd') or (char == '3'):
        return 3
    elif (char == 'E') or (char == 'e') or (char == '4'):
        return 4
    else:
        return 5


class TestScannerScore(QtWidgets.QFrame, layout.Ui_Frame):
    def __init__(self, *args, **kwargs):
        super(TestScannerScore, self).__init__(*args, **kwargs)

        self.setupUi(self)
        self.setWindowTitle('Thị giác máy tính - Ứng dụng chấm điểm thi trắc nghiệm qua hình ảnh')

        self.btn_choose.setEnabled(False)
        self.btn_score.setEnabled(False)
        self.btn_choose.clicked.connect(self.chooseImage)
        self.btn_score.clicked.connect(self.getTheScore)
        self.btn_update.clicked.connect(self.updateAnswer)

    def updateAnswer(self):
        global ANSWER_KEY
        ans_sheet = str(self.txt_input_answer.toPlainText())
        ans_sheet = ans_sheet.split(',')
        result = dict()
        for i, v in enumerate(ans_sheet):
            result[i] = changeText(v.split(':')[1].strip())
        ANSWER_KEY = result
        print(ANSWER_KEY)

        self.btn_choose.setEnabled(True)

    def chooseImage(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open File', 'c:\\', 'Images file (*.jpg *.jpeg *.png)')
        self.imagePath = fname[0]
        print(self.imagePath)
        pixmap = QtGui.QPixmap(self.imagePath)
        self.exam_img.setPixmap(pixmap)
        self.btn_score.setEnabled(True)

    def getTheScore(self):
        # tien xu lí -------------------------
        # load the image, convert it to grayscale, blur it
        # slightly, then find edges
        image = cv2.imread(self.imagePath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
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
        warped = four_point_transform(gray, docCnt.reshape(4, 2))
        # apply Otsu's thresholding method to binarize the warped
        # piece of paper
        thresh = cv2.threshold(warped, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # find contours in the thresholded image, then initialize
        # the list of contours that correspond to questions
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        questionCnts = []
        # loop over the contours
        for c in cnts:
            # compute the bounding box of the contour, then use the
            # bounding box to derive the aspect ratio
            (x, y, w, h) = cv2.boundingRect(c)
            ar = w / float(h)
            # in order to label the contour as a question, region
            # should be sufficiently wide, sufficiently tall, and
            # have an aspect ratio approximately equal to 1
            if w >= 20 and h >= 20 and 0.9 <= ar <= 1.1:
                questionCnts.append(c)

        # sorting the contours from top to bottom
        questionCnts = contours.sort_contours(questionCnts, method="top-to-bottom")[0]
        correct = 0

        final = ''
        for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
            cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
            bubbled = None

            answer = ''
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
                answer = "Đúng"
            print('ket qua lam: {}'.format(bubbled[1]))

            # draw the outline of the correct answer on the test
            cv2.drawContours(paper, [cnts[k]], -1, color, 3)
            # print('cau {}: {} - dap an dung: {}\n------------------'.format(q, bubbled[1], k))
            if answer is None or answer == '':
                answer = "Sai - Đán án: {}".format(getAnswerChar(k))
            final = final + 'Cau {}: {} - {} \n'.format(q + 1, getAnswerChar(bubbled[1]), answer)

        text = 'Tong diem: {}'.format((correct / 5.0) * 100)
        result = text + '\n\n' + final

        self.lbl_did_right.setText('{}/{}'.format(correct, len(ANSWER_KEY)))
        self.lbl_did_wrong.setText('{}/{}'.format(len(ANSWER_KEY) - correct, len(ANSWER_KEY)))
        self.lbl_total_score.setText('{}%'.format((correct / 5.0) * 100))
        self.exam_result.setPixmap(convert_cv_qt(paper))
        self.txt_result.setPlainText(result)


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    widget = TestScannerScore()
    widget.show()
    try:
        sys.exit(app.exec_())
    except (SystemError, SystemExit):
        app.exit()
