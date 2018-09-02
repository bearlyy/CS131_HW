import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

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
    kernel_new = np.flip(np.flip(kernel, 1), 0).copy()
    x = (int)((Hk - 1) / 2)
    y = (int)((Wk - 1) / 2)
    Hmin = min(Hi, Hk)
    Wmin = min(Wi, Wk)

    for n in range(Hi):
        for m in range(Wi):
            for k in range(Hi):
                for l in range(Wi):
                    if n - k + x >= Hk or m - l + y >= Wk:
                        continue
                    if n - k + x < 0 or m - l + y < 0:
                        continue
                    a = image[k][l]
                    b = kernel[n - k + x][m - l + y]
                    out[n][m] += image[k][l] * kernel[n - k + x][m - l + y]
    ### END YOUR CODE

    return out


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


def conv_fast(image, kernel):
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


def conv_faster(image, kernel):
    """
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
    pass
    ### END YOUR CODE

    return out


def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    g_flip = np.flip(np.flip(g, 0), 1)
    out = conv_fast(f, g_flip)
    ### END YOUR CODE

    return out


def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hg, Wg = g.shape
    g_mean = np.sum(g) / Hg / Wg
    g_zero_mean = g - g_mean
    g_zero_flip = np.flip(np.flip(g_zero_mean, 0), 1)
    out = conv_fast(f, g_zero_flip)
    ### END YOUR CODE

    return out


def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))

    f_pad = zero_pad(f, Hg // 2, Wg // 2)
    g_std = np.std(g)
    g_mean = g - np.mean(g)
    g_new = g_mean / g_std
    for i in range(Hf):
        for j in range(Wf):
            f_pad_std = np.std(f_pad[i:(i + Hg), j:(j + Wg)])
            f_pad_mean = f_pad[i:(i + Hg), j:(j + Wg)] - np.mean(f_pad[i:(i + Hg), j:(j + Wg)])
            f_pad_new = f_pad_mean / f_pad_std
            out[i][j] = np.sum(np.multiply(g_new, f_pad_new))
            ### END YOUR CODE

    return out
