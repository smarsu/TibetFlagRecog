import os
import cv2

pics = os.listdir("../Images")
cnt = 0
for pic in pics:
    path = os.path.join("../Images", pic)
    try:
        cvimg = cv2.imread(path)
        cv2.imshow("t", cvimg)
        h, w, _ = cvimg.shape
        if min(h, w) < 32 or w / h > 2 or w / h < 0.5:
            pic = pic.replace('&', '\&')
            print(pic)
            cnt += 1
            os.system("rm -r ../Images/" + str(pic))
            os.system("rm -r ../Annotations/" + str(pic)[:-4] + '.txt')
    except:
        pic = pic.replace('&', '\&')
        print(pic)
        os.system("rm -r ../Images/" + str(pic))
        os.system("rm -r ../Annotations/" + str(pic)[:-4] + '.txt')
print(cnt)
