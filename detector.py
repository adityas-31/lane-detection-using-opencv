import matplotlib.pylab as plt
import cv2
import numpy as np

def roi(img , vertices):
    #blank matrix that defines image height and width
    mask = np.zeros_like(img)
    # channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask , vertices , match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def draw_line(img , lines):
    img_copy = np.copy(img)
    blank_image= np.zeros((img.shape[0] , img.shape[1] , 3) , np.uint8)

    for line in lines:
        for x1 , y1 , x2 , y2 in line:
            cv2.line(blank_image, (x1 , y1) , (x2 , y2) , (0 , 255 , 0) , thickness=10)

    img = cv2.addWeighted(img , 0.8 , blank_image , 1 , 0 )
    return img
#412 , 410 , 409 , 405

paths = ['images/Screenshot (405).png' , 'images/Screenshot (409).png' , 'images/Screenshot (410).png' , 'images/Screenshot (412).png']
for path in paths:
    image = cv2.imread(path)

    # plt.imshow(image)
    # plt.show()

    original_image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    image = cv2.blur(image , (6,6))
    image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

    height = image.shape[0]
    width = image.shape[1]

    roi_vertices = [(0 , height) , (width/3 , height/2) , (2*width/3 , height/2) ,  (width , height)]

    canny_image = cv2.Canny(image , 100, 200)
    cropped_image = roi(canny_image, np.array([roi_vertices] , np.int32))

    lines = cv2.HoughLinesP(cropped_image , rho=6 , theta=np.pi/60 , threshold=160 , lines=np.array([]) , minLineLength=40 , maxLineGap=25)
    image_with_lines = draw_line(original_image , lines)
    plt.imshow(image_with_lines)
    plt.show()
