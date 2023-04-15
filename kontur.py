import cv2
import numpy
import tensorflow
from PIL import Image
from io import BytesIO

my_photo = cv2.imread('C:\\Users\\kazan\\Desktop\\Cat\\12.jpg')

scale_percent = 99
k = [600, 600]
width = int(my_photo.shape[1] * scale_percent / 100)
height = int(my_photo.shape[0] * scale_percent / 100)

while True:
    if (k[0] < width) or (k[1] < height):
        scale_percent -= 1
        width = int(my_photo.shape[1] * scale_percent / 100)
        height = int(my_photo.shape[0] * scale_percent / 100)
    else:
        break

dim =(width, height)

my_photo = cv2.resize(my_photo, dim)

filterd_image  = cv2.medianBlur(my_photo,7)
img_grey = cv2.cvtColor(filterd_image,cv2.COLOR_BGR2GRAY)

#set a thresh
thresh = 100

#get threshold image
ret,thresh_img = cv2.threshold(img_grey, thresh, 255, cv2.THRESH_BINARY)

#find contours
contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#create an empty image for contours
img_contours = numpy.uint8(numpy.zeros((my_photo.shape[0],my_photo.shape[1])))

cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)

cv2.imshow('origin', my_photo) # выводим итоговое изображение в окно
cv2.imshow('res', img_contours) # выводим итоговое изображение в окно

cv2.waitKey()
cv2.destroyAllWindows()