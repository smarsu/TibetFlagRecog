import os
import cv2

Annotation_path = ('../Annotations_val')
Image_path = ('../Images_val')

pics = os.listdir(Image_path)
cnt = 0
for pic in pics:
    path = os.path.join(Image_path, pic)
    try:
        cvimg = cv2.imread(path)
        cv2.imshow("t", cvimg)
        h, w, _ = cvimg.shape
        if min(h, w) < 32:
            print(pic)
            cnt += 1
            os.system("rm -r ../Images_val/" + str(pic))
            os.system("rm -r ../Annotations_val/" + str(pic)[:-4] + '.txt')
    except:
        print(pic)
        os.system("rm -r ../Images_val/" + str(pic))
        os.system("rm -r ../Annotations_val/" + str(pic)[:-4] + '.txt')
print(cnt)
