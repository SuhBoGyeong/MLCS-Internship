from ROI_revision import *
from ROI_extraction import *
from middle_line_via_snake import *
from jaw_separation import separate_jaws
import time

import csv
from prepare import *


bb_region=np.empty((270,4), dtype='int')
for i in range(1, 51):
    print("loading image number %d" % i)
    img_address = './images/%d.bmp' % i
    img = cv2.imread(img_address, 0)
    img_copy = copy.deepcopy(x=img)

    print('original image dimensions:', img.shape)
    t0 = time.time()
    initial_roi, initial_boundaries = extract_roi(image=img, return_result=1)
    print('initial ROI dimensions:', initial_roi.shape)
    revised_roi, revised_boundaries = revise_boundaries(image=initial_roi, return_result=1)
    print('final ROI dimensions:', revised_roi.shape)
    t1 = time.time()
    print('elapsed time for ROI extraction & revision: %.2f secs' % (t1 - t0))

    upper_height = initial_boundaries[3] + revised_boundaries[3]
    left_width = initial_boundaries[0] + revised_boundaries[0]
    lower_height = upper_height + revised_roi.shape[0]
    right_width = left_width + revised_roi.shape[1]

    top_left_corner = (left_width, upper_height)
    top_right_corner = (right_width, upper_height)
    bottom_left_corner = (left_width, lower_height)
    bottom_right_corner = (right_width, lower_height)

    # print('roi points:', top_left_corner, top_right_corner, bottom_left_corner, bottom_right_corner)
    cv2.rectangle(img_copy, top_left_corner, bottom_right_corner, 0, 7)

    # plt.imshow(X=img_copy, cmap='gray')
    # plt.show()

    # print("save & continue? (y/n)")
    # if input() != 'y':
    #     print("process terminated!")
    #     exit()


    ####crop image
    y=upper_height
    x=left_width
    h=lower_height-y
    w=right_width-x

    cropped_img=img[int(y*0.8):int((y+h)*1.5),int(x*0.5):int((x+w)*1.2)]
    bb_region[i]=[int(y*0.8),int((y+h)*1.5),int(x*0.5),int((x+w)*1.2)]
    cropped_img=cv2.resize(cropped_img, dsize=(0,0), fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    #cv2.imshow('a',cropped_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite('./auto-cropped-images/%d.bmp' % i, cropped_img)

f= open('train.csv', 'w')
writer=csv.writer(f)
for i in range(0,270):
    #print(image_number[i])
    new_path=os.path.join(path,str(image_number[i])+'.bmp')
    writer.writerow([new_path, int(region[i][0]),int(region[i][1]),int(region[i][2]),int(region[i][3]),'prosthesis'])



