import numpy as np


def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    left = np.zeros((H, pad_width))
    upside = np.zeros((pad_height, 2 * pad_width + W))
    temp = np.c_[image, left]
    temp2 = np.c_[left, temp]
    temp3 = np.r_[temp2, upside]
    out = np.r_[upside, temp3]
    ### END YOUR CODE
    return out


def conv(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    image_pad = zero_pad(image, Hk // 2, Wk // 2)
    kernel_flip = np.flip(np.flip(kernel, 0), 1)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(np.multiply(kernel_flip, image_pad[i:(i + Hk), j:(j + Wk)]))

    ### END YOUR CODE

    return out


def conv_f(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0, pad_width0), (pad_width1, pad_width1))
    padded = np.pad(image, pad_width, mode='constant', constant_values=0)

    ### YOUR CODE HERE
    kernel_flip = np.flip(np.flip(kernel, 0), 1)
    for i in range(Hi):
        for j in range(Wi):
            out[i][j] = np.sum(np.multiply(kernel_flip, padded[i:(i + Hk), j:(j + Wk)]))
    ### END YOUR CODE

    return out


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x)

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel
        H, W: size of the image
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)]
    """
    neighbors = []

    for i in (y - 1, y, y + 1):
        for j in (x - 1, x, x + 1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors


def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp

    Args:
        size: int of the size of output matrix
        sigma: float of sigma to calculate kernel

    Returns:
        kernel: numpy array of shape (size, size)
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    k = (size - 1) / 2
    for i in range(size):
        for j in range(size):
            kernel[i][j] = np.exp(((i - k) ** 2 + (j - k) ** 2) / -2 / np.square(sigma)) / (
                    2 * np.pi * np.square(sigma))
    ### END YOUR CODE

    return kernel


def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: x-derivative image
    """

    out = None

    ### YOUR CODE HERE

    Hi, Wi = img.shape
    out = np.zeros([Hi, Wi])
    for i in range(Hi):
        for j in range(Wi):
            if j == 0:
                out[i, j] = img[i, j + 1] / 2
            elif j == Wi - 1:
                out[i, j] = -img[i, j - 1] / 2
            else:
                out[i, j] = (img[i, j + 1] - img[i, j - 1]) / 2

    return out


def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W)
    Returns:
        out: y-derivative image
    """

    out = None

    ### YOUR CODE HERE

    Hi, Wi = img.shape
    out = np.zeros([Hi, Wi])
    for i in range(Hi):
        for j in range(Wi):
            if i == 0:
                out[i, j] = img[i + 1, j] / 2
            elif i == Hi - 1:
                out[i, j] = -img[i - 1, j] / 2
            else:
                out[i, j] = (img[i + 1, j] - img[i - 1, j]) / 2

    ### END YOUR CODE

    return out


def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W)

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W)
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W)
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    p_x = partial_x(img)
    p_y = partial_y(img)
    G = np.sqrt(np.square(p_x) + np.square(p_y))
    theta = np.arctan2(p_y, p_x)
    theta = np.where(theta >= 0, theta, theta + 2 * np.pi)
    theta = theta / np.pi * 180
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W)
        theta: direction of gradients with shape of (H, W)

    Returns:
        out: non-maxima suppressed image
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45

    ### BEGIN YOUR CODE

    G_pad = np.pad(G, 1, mode='constant', constant_values=0)
    theta_pad = np.pad(theta, 1, mode='constant', constant_values=0)
    H, W = G_pad.shape
    out = np.zeros((H, W))

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            if theta_pad[i][j] == 0 or theta_pad[i][j] == 180 or theta_pad[i][j] == 360:
                if G_pad[i][j] >= G_pad[i][j + 1] and G_pad[i][j] >= G_pad[i][j - 1]:
                    out[i][j] = G_pad[i][j]
            if theta_pad[i][j] == 90 or theta_pad[i][j] == 270:
                if G_pad[i][j] >= G_pad[i + 1][j] and G_pad[i][j] >= G_pad[i - 1][j]:
                    out[i][j] = G_pad[i][j]
            if theta_pad[i][j] == 45 or theta_pad[i][j] == 225:
                if G_pad[i][j] >= G_pad[i + 1][j + 1] and G_pad[i][j] >= G_pad[i - 1][j - 1]:
                    out[i][j] = G_pad[i][j]
            if theta_pad[i][j] == 135 or theta_pad[i][j] == 315:
                if G_pad[i][j] >= G_pad[i - 1][j + 1] and G_pad[i][j] >= G_pad[i + 1][j - 1]:
                    out[i][j] = G_pad[i][j]
    out = out[1:H - 1, 1:W - 1]
    ### END YOUR CODE

    return out


def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response
        high: high threshold(float) for strong edges
        low: low threshold(float) for weak edges

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values above
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values below the
            higher threshould and above the lower threshold.
    """

    strong_edges = np.zeros(img.shape)
    weak_edges = np.zeros(img.shape)

    ### YOUR CODE HERE
    temp_edges = np.where(img >= low, img, 0)
    strong_edges = np.where(img >= high, 1, 0)
    weak_edges = np.where(temp_edges < high, 0.5, 0)
    ### END YOUR CODE

    return strong_edges, weak_edges


def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W)
        weak_edges: binary image of shape (H, W)
    Returns:
        edges: numpy array of shape(H, W)
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W))

    ### YOUR CODE HERE
    # indices = np.stack(np.nonzero(weak_edges)).T
    edges = strong_edges.copy()
    for i in indices:
        neighbors = get_neighbors(i[0], i[1], H, W)
        for j in neighbors:
            if weak_edges[j[0]][j[1]] > 0:
                edges[j[0]][j[1]] = weak_edges[j[0]][j[1]]
    ### END YOUR CODE

    return edges


def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W)
        kernel_size: int of size for kernel matrix
        sigma: float for calculating kernel
        high: high threshold for strong edges
        low: low threashold for weak edges
    Returns:
        edge: numpy array of shape(H, W)
    """
    ### YOUR CODE HERE

    kernel = gaussian_kernel(kernel_size, sigma)
    smoothed = conv(img, kernel)
    G, theta = gradient(smoothed)
    nms = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge


def hough_transform(img):
    """ Transform points in the input image into Hough space.

    Use the parameterization:
        rho = x * cos(theta) + y * sin(theta)
    to transform a point (x,y) to a sine-like function in Hough space.

    Args:
        img: binary image of shape (H, W)

    Returns:
        accumulator: numpy array of shape (m, n)
        rhos: numpy array of shape (m, )
        thetas: numpy array of shape (n, )
    """
    # Set rho and theta ranges
    W, H = img.shape
    diag_len = int(np.ceil(np.sqrt(W * W + H * H)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0 + 1)
    thetas = np.deg2rad(np.arange(-90.0, 90.0))

    # Cache some reusable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Initialize accumulator in the Hough space
    accumulator = np.zeros((2 * diag_len + 1, num_thetas), dtype=np.uint64)
    ys, xs = np.nonzero(img)
    
    
    # Transform each point (x, y) in image
    # Find rho corresponding to values in thetas
    # and increment the accumulator in the corresponding coordiate.
    ### YOUR CODE HERE
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        for j in range(num_thetas):
            index = int(x * cos_t[j] + y * sin_t[j] + diag_len)
            accumulator[index, j] += 1
            rhos[index] = x * cos_t[j] + y * sin_t[j]
    ### END YOUR CODE

    return accumulator, rhos, thetas
