import cv2
import numpy as np
import classe_rede_neural as nnc
import MLP_MNIST_2 as MLP

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

a1 = nnc.load_neural_network('neural_network2.xlsx')

img = np.zeros((140,140,3), np.uint8)
# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=10)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(img,(pt1_x,pt1_y),(x,y),color=(255,255,255),thickness=10)
        recog()
    elif event == cv2.EVENT_RBUTTONUP:
        #img = np.zeros((140,140, 3), np.uint8)
        # cv2.rectangle(img=img, pt1=(0,0),pt2=(140,140),color=(0, 0, 0))
        cv2.line(img, (1, 1), (0, 0), color=(0, 0, 0), thickness=1000)
        #erase()
        print("f")


cv2.namedWindow('test draw')
cv2.setMouseCallback('test draw',line_drawing)
def erase():
    img = np.zeros((140, 140, 3), np.uint8)
    cv2.circle(img, (int(140 / 2), int(128 / 2)), 10, (0, 0, 0), -1)
    print("f")

def recog():
    print("Enter")
    # print(img)
    img_resize = cv2.resize(src=img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("im_resize", img_resize)
    # y = a1.forward_propagation(img_resize)
    cv2.imwrite("im_resize.png", img_resize)
    img_resize_rede = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    # print(np.shape(img_resize))
    img_resize_rede = np.reshape(img_resize_rede, (28 * 28, 1))
    img_resize_rede = img_resize_rede / 255 * 2. - 1.
    num_out, y_out = MLP.digit_recog(a1, img_resize_rede)
    print(num_out)
    print(y_out)

while(1):
    cv2.imshow('test draw',img)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    if cv2.waitKey(1) & 0xFF == 13:
        recog()

    if cv2.waitKey(1) & 0xFF == ord('f'):
        print("f")
        img = np.zeros((140, 140, 3), np.uint8)
        #cv2.rectangle(img=img, pt1=(0,0),pt2=(140,140),color=(255,255, 25))


cv2.destroyAllWindows()