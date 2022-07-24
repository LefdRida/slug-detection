# Measuring the height of moving slugs.

This project is the application of image processing techniques in hydraulic field.

The aim is to measure the height of moving slugs in a video by applying segmentation techniques. We have applied Otsu thresholding method to segment the video frames.

## Steps:
 
  - We keep only the region of interest (the tube where the slugs move):
   
   ![alt text](https://github.com/LefdRida/slug-detection/blob/main/images/cropped.jpg)
   
   - We segment the image using Otsu thesholding:
   
   ![alt text](https://github.com/LefdRida/slug-detection/blob/main/images/treshholded.bmp)
   
   - We apply opening operation to remove some noise belonging to the tube region
   
   ![alt text](https://github.com/LefdRida/slug-detection/blob/main/images/opentreshholded.bmp)

## Result

The final result is the following spatiotemporal image which show how the height of each slug, in the video, is changing during its passage through the tube:

<img align="center" src="https://github.com/LefdRida/slug-detection/blob/main/images/resultvid1.jpg" height="400" width="150"/>
