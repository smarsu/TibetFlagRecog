import os
import cv2

Annotation_path = ('../../Annotations')
Image_path = ('../../Images')


f = [os.path.join(root, name) for root, dirs, files in os.walk('.') \
    for name in files]

class_with_picname = [(_f.split('/')[1], _f.split('/')[-1][:-4]) for _f in f]

f_with_class_with_picname = zip(f, class_with_picname)

for p, t in f_with_class_with_picname:
    print(p)
    clsf, pic_name = t
    cvimg = cv2.imread(p)
    cv2.imwrite(os.path.join(Image_path, pic_name + '.jpg'), cvimg)
    with open(os.path.join(Annotation_path, pic_name + '.txt'), 'w') as f:
        f.write(str(clsf).replace('_', '\n') + '\n')  # File name just like 'gun_person', if one pic own many category.

if __name__ == "__main__":
    for a in f_with_class_with_picname:
        print(a)
