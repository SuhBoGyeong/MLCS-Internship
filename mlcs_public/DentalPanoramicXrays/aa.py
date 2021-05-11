import cv2

img1=cv2.imread('./Images/11.png')
img2=cv2.imread('./Images/13.png')

img1=cv2.resize(img1, dsize=(0,0), fx=1/8,fy=1/8,interpolation=cv2.INTER_LINEAR)
img2=cv2.resize(img2, dsize=(0,0), fx=1/8,fy=1/8,interpolation=cv2.INTER_LINEAR)

cv2.imwrite('test11.jpg',img1)
cv2.imwrite('test13.jpg',img2)