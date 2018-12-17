# imageProcess

* Package 'imageProcess' can process images, which mainly used for computer vision

## data_augmentation

> For use this, we need numpy and cv2

* It collection many methons to do data augmentation. As follows:
    
    * images_mean_std():
        * Calculate the mean and std with all images' all pixels in dataset. (Not calculated by channel)
        * Args:
            images: 