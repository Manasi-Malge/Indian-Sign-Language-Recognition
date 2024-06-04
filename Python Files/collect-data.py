import os
import string
import cv2

# Create the directory structure
if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data"):
    os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data")
if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train"):
    os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train")
if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test"):
    os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test")
for i in range(10):
    if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train/" + str(i)):
        os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train/" + str(i))
    if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test/" + str(i)):
        os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test/" + str(i))

for i in string.ascii_uppercase:
    if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train/" + i):
        os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/train/" + i)
    if not os.path.exists("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test/" + i):
        os.makedirs("C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/test/" + i)

# Train or test 
mode = 'train'
directory = 'C:/Users/User/OneDrive/Desktop/Sign-Language-to-Text/data/' + mode + '/'
minValue = 70

cap = cv2.VideoCapture(0)
interrupt = -1

while True:
    _, frame = cap.read()
    # Simulating mirror image
    frame = cv2.flip(frame, 1)

    # Getting count of existing images
    count = {
        '0': len(os.listdir(directory + "/0")),
        '1': len(os.listdir(directory + "/1")),
        '2': len(os.listdir(directory + "/2")),
        '3': len(os.listdir(directory + "/3")),
        '4': len(os.listdir(directory + "/4")),
        '5': len(os.listdir(directory + "/5")),
        '6': len(os.listdir(directory + "/6")),
        '7': len(os.listdir(directory + "/7")),
        '8': len(os.listdir(directory + "/8")),
        '9': len(os.listdir(directory + "/9")),
        'a': len(os.listdir(directory + "/A")),
        'b': len(os.listdir(directory + "/B")),
        'c': len(os.listdir(directory + "/C")),
        'd': len(os.listdir(directory + "/D")),
        'e': len(os.listdir(directory + "/E")),
        'f': len(os.listdir(directory + "/F")),
        'g': len(os.listdir(directory + "/G")),
        'h': len(os.listdir(directory + "/H")),
        'i': len(os.listdir(directory + "/I")),
        'j': len(os.listdir(directory + "/J")),
        'k': len(os.listdir(directory + "/K")),
        'l': len(os.listdir(directory + "/L")),
        'm': len(os.listdir(directory + "/M")),
        'n': len(os.listdir(directory + "/N")),
        'o': len(os.listdir(directory + "/O")),
        'p': len(os.listdir(directory + "/P")),
        'q': len(os.listdir(directory + "/Q")),
        'r': len(os.listdir(directory + "/R")),
        's': len(os.listdir(directory + "/S")),
        't': len(os.listdir(directory + "/T")),
        'u': len(os.listdir(directory + "/U")),
        'v': len(os.listdir(directory + "/V")),
        'w': len(os.listdir(directory + "/W")),
        'x': len(os.listdir(directory + "/X")),
        'y': len(os.listdir(directory + "/Y")),
        'z': len(os.listdir(directory + "/Z"))
    }

    # Printing the count in each set to the screen
    # cv2.putText(frame, "MODE : "+mode, (10, 50), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    # cv2.putText(frame, "IMAGE COUNT", (10, ), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    '''cv2.putText(frame, "0 : " + str(count['0']), (10, 70), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "1 : " + str(count['1']), (10, 80), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "2 : " + str(count['2']), (10, 90), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "3 : " + str(count['3']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "4 : " + str(count['4']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "5 : " + str(count['5']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "6 : " + str(count['6']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "7 : " + str(count['7']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "8 : " + str(count['8']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "9 : " + str(count['9']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "a : " + str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "b : " + str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "c : " + str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "d : " + str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "e : " + str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "f : " + str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "g : " + str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "h : " + str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "i : " + str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "k : " + str(count['k']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "l : " + str(count['l']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "m : " + str(count['m']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "n : " + str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "o : " + str(count['o']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "p : " + str(count['p']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "q : " + str(count['q']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "r : " + str(count['r']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "s : " + str(count['s']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "t : " + str(count['t']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "u : " + str(count['u']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "v : " + str(count['v']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "w : " + str(count['w']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "x : " + str(count['x']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "y : " + str(count['y']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
    cv2.putText(frame, "z : " + str(count['z']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)'''
    # Coordinates of the ROI
    x1 = int(0.5 * frame.shape[1])
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = int(0.5 * frame.shape[1])
    # Drawing the ROI
    # The increment/decrement by 1 is to compensate for the bounding box
    cv2.rectangle(frame, (220 - 1, 9), (620 + 1, 419), (255, 0, 0), 1)
    # Extracting the ROI
    roi = frame[10:410, 220:520]
    #    roi = cv2.resize(roi, (64, 64))

    cv2.imshow("Frame", frame)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # #blur = cv2.bilateralFilter(roi,9,75,75)

    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # time.sleep(5)
    # cv2.imwrite("/home/rc/Downloads/soe/im1.jpg", roi)
    # test_image = func("/home/rc/Downloads/soe/im1.jpg")

    test_image = cv2.resize(test_image, (600, 400))
    cv2.imshow("test", test_image)

    # _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(mask, kernel, iterations=1)
    # img = cv2.erode(mask, kernel, iterations=1)
    # do the processing after capturing the image!
    #    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #    _, roi = cv2.threshold(roi, 120, 255, cv2.THRESH_BINARY)

    #  gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("GrayScale", gray)
    # blur = cv2.GaussianBlur(gray,(5,5),2)

    # blur = cv2.bilateralFilter(roi,9,75,75)

    # th3 = cv2.adaptiveThreshold(frame,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    # ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # cv2.imshow("ROI", roi)
    # roi = frame
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # esc key
        break
    if interrupt & 0xFF == ord('0'):
        cv2.imwrite(directory + '0/' + str(count['0']) + '.jpg', roi)
    if interrupt & 0xFF == ord('1'):
        cv2.imwrite(directory + '1/' + str(count['1']) + '.jpg', roi)
    if interrupt & 0xFF == ord('2'):
        cv2.imwrite(directory + '2/' + str(count['2']) + '.jpg', roi)
    if interrupt & 0xFF == ord('3'):
        cv2.imwrite(directory + '3/' + str(count['3']) + '.jpg', roi)
    if interrupt & 0xFF == ord('4'):
        cv2.imwrite(directory + '4/' + str(count['4']) + '.jpg', roi)
    if interrupt & 0xFF == ord('5'):
        cv2.imwrite(directory + '5/' + str(count['5']) + '.jpg', roi)
    if interrupt & 0xFF == ord('6'):
        cv2.imwrite(directory + '6/' + str(count['6']) + '.jpg', roi)
    if interrupt & 0xFF == ord('7'):
        cv2.imwrite(directory + '7/' + str(count['7']) + '.jpg', roi)
    if interrupt & 0xFF == ord('8'):
        cv2.imwrite(directory + '8/' + str(count['8']) + '.jpg', roi)
    if interrupt & 0xFF == ord('9'):
        cv2.imwrite(directory + '9/' + str(count['9']) + '.jpg', roi)
    if interrupt & 0xFF == ord('a'):
        cv2.imwrite(directory + 'A/' + str(count['a']) + '.jpg', roi)
    if interrupt & 0xFF == ord('b'):
        cv2.imwrite(directory + 'B/' + str(count['b']) + '.jpg', roi)
    if interrupt & 0xFF == ord('c'):
        cv2.imwrite(directory + 'C/' + str(count['c']) + '.jpg', roi)
    if interrupt & 0xFF == ord('d'):
        cv2.imwrite(directory + 'D/' + str(count['d']) + '.jpg', roi)
    if interrupt & 0xFF == ord('e'):
        cv2.imwrite(directory + 'E/' + str(count['e']) + '.jpg', roi)
    if interrupt & 0xFF == ord('f'):
        cv2.imwrite(directory + 'F/' + str(count['f']) + '.jpg', roi)
    if interrupt & 0xFF == ord('g'):
        cv2.imwrite(directory + 'G/' + str(count['g']) + '.jpg', roi)
    if interrupt & 0xFF == ord('h'):
        cv2.imwrite(directory + 'H/' + str(count['h']) + '.jpg', roi)
    if interrupt & 0xFF == ord('i'):
        cv2.imwrite(directory + 'I/' + str(count['i']) + '.jpg', roi)
    if interrupt & 0xFF == ord('j'):
        cv2.imwrite(directory + 'J/' + str(count['j']) + '.jpg', roi)
    if interrupt & 0xFF == ord('k'):
        cv2.imwrite(directory + 'K/' + str(count['k']) + '.jpg', roi)
    if interrupt & 0xFF == ord('l'):
        cv2.imwrite(directory + 'L/' + str(count['l']) + '.jpg', roi)
    if interrupt & 0xFF == ord('m'):
        cv2.imwrite(directory + 'M/' + str(count['m']) + '.jpg', roi)
    if interrupt & 0xFF == ord('n'):
        cv2.imwrite(directory + 'N/' + str(count['n']) + '.jpg', roi)
    if interrupt & 0xFF == ord('o'):
        cv2.imwrite(directory + 'O/' + str(count['o']) + '.jpg', roi)
    if interrupt & 0xFF == ord('p'):
        cv2.imwrite(directory + 'P/' + str(count['p']) + '.jpg', roi)
    if interrupt & 0xFF == ord('q'):
        cv2.imwrite(directory + 'Q/' + str(count['q']) + '.jpg', roi)
    if interrupt & 0xFF == ord('r'):
        cv2.imwrite(directory + 'R/' + str(count['r']) + '.jpg', roi)
    if interrupt & 0xFF == ord('s'):
        cv2.imwrite(directory + 'S/' + str(count['s']) + '.jpg', roi)
    if interrupt & 0xFF == ord('t'):
        cv2.imwrite(directory + 'T/' + str(count['t']) + '.jpg', roi)
    if interrupt & 0xFF == ord('u'):
        cv2.imwrite(directory + 'U/' + str(count['u']) + '.jpg', roi)
    if interrupt & 0xFF == ord('v'):
        cv2.imwrite(directory + 'V/' + str(count['v']) + '.jpg', roi)
    if interrupt & 0xFF == ord('w'):
        cv2.imwrite(directory + 'W/' + str(count['w']) + '.jpg', roi)
    if interrupt & 0xFF == ord('x'):
        cv2.imwrite(directory + 'X/' + str(count['x']) + '.jpg', roi)
    if interrupt & 0xFF == ord('y'):
        cv2.imwrite(directory + 'Y/' + str(count['y']) + '.jpg', roi)
    if interrupt & 0xFF == ord('z'):
        cv2.imwrite(directory + 'Z/' + str(count['z']) + '.jpg', roi)

cap.release()
cv2.destroyAllWindows()
