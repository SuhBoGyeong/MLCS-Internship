import csv
from prepare import *

idx=0
path='train'
for images in images_data:
    img=cv2.imread(images,0)
    img=cv2.resize(img, dsize=(0,0), fx=1/8,fy=1/8,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(path,'{}.bmp'.format(idx)),img)
    idx+=1

f= open('bb_train.csv', 'w')
writer=csv.writer(f)
for i in range(0,270):
    #print(image_number[i])
    new_path=os.path.join(path,str(image_number[i])+'.bmp')
    writer.writerow([new_path, int(region[i][0]),int(region[i][1]),int(region[i][2]),int(region[i][3]),'prosthesis'])








