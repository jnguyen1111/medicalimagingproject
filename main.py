import torch
import torchio as tio
import matplotlib.pyplot as plt
import numpy as np
import phantominator
from phantominator import shepp_logan
import imgaug.augmenters as iaa

def showImage(image):
    plt.title('Image')
    plt.imshow(image, cmap='gray')
    plt.show()

#return a vertically flipped image using the original image
def verticalRotation(image):
    vflip_obj = iaa.Flipud(p=1)
    vertical_flipped_image  = vflip_obj.augment_image(image)
    return vertical_flipped_image

#return a horizontally flipped image using the original image
def horizontalRotation(image):
    hflip_obj = iaa.Fliplr(p=1)
    horizontal_flipped_image = hflip_obj.augment_image(image)
    return horizontal_flipped_image

#return a image that has been flipped with specified degree user requests
def imageRotation(degree,image):
    rotation_object =iaa.Affine(rotate=(degree))
    rotated_image = rotation_object.augment_image(image)
    return rotated_image

#returns a image with deformations applied to it
def elasticDeformation():
    ...

def main():
    M0, T1, T2 = shepp_logan((256, 256, 1), MR=True, zlims=(-.25, .25))
    vert_image_flip = verticalRotation(M0)
    horz_image_flip = horizontalRotation(M0)
    image_rotation = imageRotation(200,M0)
    print("showing regular image\n")
    showImage(M0)
    print("showing vertical image\n")
    showImage(vert_image_flip)
    print("showing horizontal image\n")
    showImage(horz_image_flip)
    print("showing rotated image\n")
    showImage(image_rotation)

if __name__ == "__main__":
    main()
