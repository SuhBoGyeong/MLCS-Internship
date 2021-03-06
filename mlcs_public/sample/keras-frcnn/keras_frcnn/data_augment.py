import cv2
import numpy as np
import copy
from clahe import clahe

crop_ratio=0.8

def augment(img_data, config, augment=True):
	assert 'filepath' in img_data
	assert 'bboxes' in img_data
	assert 'width' in img_data
	assert 'height' in img_data

	img_data_aug = copy.deepcopy(img_data)

	img = cv2.imread(img_data_aug['filepath'])
	###########clahe############
	#img=clahe(img)
	#cv2.imshow('a',img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

	if augment:
		rows, cols = img.shape[:2]

		if config.use_horizontal_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 1)
			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug['bboxes']:
				y1 = bbox['y1']
				y2 = bbox['y2']
				bbox['y2'] = rows - y1
				bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass
		
		if config.crop:
			n_r=int(rows*crop_ratio)
			n_c=int(cols*crop_ratio)
			y=np.random.randint(0, rows-n_r) 
			x=np.random.randint(0, cols-n_c)
			#print(y,x)
			img = img[y:y+n_r,x:x+n_c]

			for bbox in img_data_aug['bboxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']				
				bbox['x1'] = -x+x1
				bbox['x2'] = -x+x2
				bbox['y1'] = -y+y1 
				bbox['y2'] = -y+y2 
				#print(bbox['x1'],bbox['x2'],bbox['y1'],bbox['y2'])
				#cv2.rectangle(img,(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0,0,255),2)
				#cv2.imshow('a',img)
				#cv2.waitKey(0) 
				#cv2.destroyAllWindows()
				#if (bbox['x2']>n_c) or (bbox['y2']>n_r):
					#pass

		if config.clahe and np.random.randint(0, 2) == 0:
			gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
			clahe=cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			img2=clahe.apply(gray)
			img=cv2.cvtColor(img2,cv2.COLOR_GRAY2RGB)
			#cv2.imshow('a',img)
			#cv2.waitKey(0)
			#cv2.destroyAllWindows()

	img_data_aug['width'] = img.shape[1]
	img_data_aug['height'] = img.shape[0]
	return img_data_aug, img
