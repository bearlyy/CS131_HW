import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
from skimage import color
from skimage import io

def load(image_path):
    """ Loads an image from a file path

    Args:
        image_path: file path to the image

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    # Use skimage io.imread
    out = io.imread(image_path)
    ### END YOUR CODE

    return out


def change_value(image):
    """ Change the value of every pixel by following x_n = 0.5*x_p^2 
        where x_n is the new value and x_p is the original value

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    out = 0.5 * (image ** 2)
    ### END YOUR CODE

    return out


def convert_to_grey_scale(image):
    """ Change image to gray scale

    Args:
        image: numpy array of shape(image_height, image_width, 3)

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    out = None

    ### YOUR CODE HERE
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    out = 0.2989 * r + 0.5870 * g + 0.1140 * b
    ### END YOUR CODE

    return out

def rgb_decomposition(image, channel):
    """ Return image **excluding** the rgb channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None

    ### YOUR CODE HERE
    if channel == 'R':
        out = image[:,:,0] + image[:,:,1]
    elif channel == 'G':
        out = image[:,:,0] + image[:,:,2]
    elif channel == 'B':
        out = image[:,:,1] + image[:,:,2]
    ### END YOUR CODE

    return out

def lab_decomposition(image, channel):
    """ Return image decomposed to just the lab channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    lab = color.rgb2lab(image)
    out = None

    ### YOUR CODE HERE
    if channel == 'L':
        out = lab[:,:,0]
    elif channel == 'A':
        out = lab[:,:,1]
    elif channel == 'B':
        out = lab[:,:,2]
    ### END YOUR CODE

    return out

def hsv_decomposition(image, channel='H'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    hsv = color.rgb2hsv(image)
    out = None

    ### YOUR CODE HERE
    if channel == 'H':
        out = hsv[:,:,0]
    elif channel == 'S':
        out = hsv[:,:,1]
    elif channel == 'V':
        out = hsv[:,:,2]
    ### END YOUR CODE

    return out

def rgb_decomposition_new(image, channel='R'):
    """ Return image decomposed to just the hsv channel specified

    Args:
        image: numpy array of shape(image_height, image_width, 3)
        channel: str specifying the channel

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """
    
    out = None

    ### YOUR CODE HERE
    if channel == 'R':
        out = image[:,:,0]
    elif channel == 'G':
        out = image[:,:,1]
    elif channel == 'B':
        out = image[:,:,2]
    ### END YOUR CODE

    return out

def mix_images(image1, image2, channel1, channel2):
    """ Return image which is the left of image1 and right of image 2 excluding
    the specified channels for each image

    Args:
        image1: numpy array of shape(image_height, image_width, 3)
        image2: numpy array of shape(image_height, image_width, 3)
        channel1: str specifying channel used for image1
        channel2: str specifying channel used for image2

    Returns:
        out: numpy array of shape(image_height, image_width, 3)
    """

    out = None
    ### YOUR CODE HERE
    if channel1 == 'R' or channel1 == 'G' or channel1 == 'B':
        left = rgb_decomposition_new(image1,channel1)
    elif channel1 == 'H' or channel1 == 'S' or channel1 == 'V':
        left = hsv_decomposition(image1,channel1)
    elif channel1 == 'L' or channel1 == 'A' :
        left = lab_decomposition(image1,channel1)
        
    if channel2 == 'R' or channel2 == 'G' or channel2 == 'B':
        right = rgb_decomposition_new(image2,channel2)
    elif channel2 == 'H' or channel2 == 'S' or channel2 == 'V':
        right = hsv_decomposition(image2,channel2)
    elif channel2 == 'L' or channel2 == 'A' :
        right = lab_decomposition(image2,channel2)
    
    m,n = left.shape
    print(left[:,:n//2].shape)
    print(right[:,n//2:].shape)
    out = np.concatenate((left[:,:n//2],right[:,n//2:]),axis=1)
    ### END YOUR CODE

    return out
