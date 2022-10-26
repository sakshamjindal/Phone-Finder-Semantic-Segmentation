# useage
test:
``` python find_phone.py path_to_img ```
print the x y coordinate of the phone, return -1 -1 if no phone is found in the image

train:
``` python train_phone_finder.py pah_to_dataset_folder ```

# Description
Firstly, a KernelRidge model is used to segment the image into phone area and other area. The image is converted to HSV space for lighting invariance. Then minArea algorithm from OpenCV is used to find the rotated bounding box of the phone area. Finally, width-height ratio and area are used to determine if the bounding box is a phone. I assumed there is only one phone in the given image. This approach achieves 82.2% accuracy on given dataset.  

# Next Steps

Based on this approach, a classification model could be used in the final step to determine if the bounding box is a phone. Moreover, if we have more well labeled data, some CNN based method like Mask-RCNN could be trained. They are much more reliable and also fast with GPU. 

# Ways to improve the data collection
Rather than just labeling the center points of the phone, labeling a rotated bounding box of the whole phone area will be helpful.