# import the necessary packages
from collections import namedtuple
import numpy as np
import cv2
import csv
from io import StringIO
 
# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

pred = {}

with open('ssd_resnet50_2_results/test_ssd_resnet50_2.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        try:
            pred['datasets/test/' + row[0] + '.jpg'] = [int(row[4]), int(row[5]), int(row[6]), int(row[7])]
        except:
            pass

gt = {}

with open('test_groundtruth.csv', newline='') as csv2:
    reader = csv.reader(csv2, delimiter=',', quotechar='|')
    for row in reader:
        try:
            gt[row[0]] = [int(row[1]), int(row[2]), int(row[3]), int(row[4])]
        except:
            pass

tuples = []

for p in pred.keys():
    if p in gt.keys():
        print(gt[p])
        print(pred[p])
        tuples.append(Detection(p, gt[p], pred[p]))        

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou

'''
# define the list of example detections
examples = [
	Detection("image_0002.jpg", [39, 63, 203, 112], [54, 66, 198, 114]),
	Detection("image_0016.jpg", [49, 75, 203, 125], [42, 78, 186, 126]),
	Detection("image_0075.jpg", [31, 69, 201, 125], [18, 63, 235, 135]),
	Detection("image_0090.jpg", [50, 72, 197, 121], [54, 72, 198, 120]),
	Detection("image_0120.jpg", [35, 51, 196, 110], [36, 60, 180, 108])]
'''
for detection in tuples:
    # load the image
    image = cv2.imread(detection.image_path)
 
    # draw the ground-truth bounding box along with the predicted
    # bounding box
    cv2.rectangle(image, tuple(detection.gt[:2]), 
    	tuple(detection.gt[2:]), (0, 255, 0), 2)
    cv2.rectangle(image, tuple(detection.pred[:2]), 
    	tuple(detection.pred[2:]), (0, 0, 255), 2)
 
    # compute the intersection over union and display it
    iou = bb_intersection_over_union(detection.gt, detection.pred)
    cv2.putText(image, "IoU: {:.4f}".format(iou), (10, 30),
    	cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    print("{}: {:.4f}".format(detection.image_path, iou))
    
    with open('ssd_resnet50_iou_list.csv', 'a') as f:
        writer = csv.writer(f, delimiter=',', quotechar='|')
        writer.writerow([detection.image_path, iou])

    # show the output image
    #cv2.imshow("Image", image)
    #cv2.waitKey(0)

