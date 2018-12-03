import cv2


def putText(cvimg, text, point=(0, 30)):
    """Put text on cvimg"""
    cvimg = cv2.putText(
        cvimg, 
        text, 
        point,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 0),
    )
    return cvimg
