from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import cv2


def get_steer_matrix_from_points(x_1, y_1, x_2, y_2):
        m = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - m * x_1

        A = y_2 - y_1
        B = x_1 - x_2
        C = x_2 * y_1 - x_1 * y_2

        return np.array([[(A * i + B * j + C ) for j in range(size_y)] for i in range(size_x)])


STORED_MATRIX_L = {}
def get_steer_matrix_left_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:              The shape of the steer matrix.

    Return:
        steer_matrix_left:  The steering (angular rate) matrix for reactive control
                            using the masked left lane markings (numpy.ndarray)
    """

    if shape in STORED_MATRIX_L:
        return STORED_MATRIX_L[shape]

    else:
        size_x, size_y = shape
        x_1, y_1 = size_x, 0
        x_2, y_2 = size_x // 3, size_y //2

        left_steer_matrix = get_steer_matrix_from_points(x_1, y_1, x_2, y_2)
        left_mask = np.array([[1 if i > size_x // 3 and j < size_y // 2 else 0 for j in range(size_y)] for i in range(size_x)])

        left_matrix = left_matrix * left_mask
        STORED_MATRIX_L[shape] = left_matrix
        return left_matrix


STORED_MATRIX_R = {}
def get_steer_matrix_right_lane_markings(shape: Tuple[int, int]) -> np.ndarray:
    """
    Args:
        shape:               The shape of the steer matrix.

    Return:
        steer_matrix_right:  The steering (angular rate) matrix for reactive control
                             using the masked right lane markings (numpy.ndarray)
    """

    # TODO: implement your own solution here
    if shape in STORED_MATRIX_R:
        return STORED_MATRIX_R[shape]
    else:
        size_x, size_y = shape
        x_1, y_1 = size_x, 0
        x_2, y_2 = size_x // 3, size_y //2

        right_steer_matrix = get_steer_matrix_from_points(x_1, y_1, x_2, y_2)
        right_mask = np.array([[1 if i > size_x // 3 and j < size_y // 2 else 0 for j in range(size_y)] for i in range(size_x)])

        right_matrix = right_matrix * right_mask
        STORED_MATRIX_R[shape] = right_matrix
        return right_matrix



def detect_lane_markings(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        image: An image from the robot's camera in the BGR color space (numpy.ndarray)
    Return:
        mask_left_edge:   Masked image for the dashed-yellow line (numpy.ndarray)
        mask_right_edge:  Masked image for the solid-white line (numpy.ndarray)
    """
    h, w, _ = img_bgr.shape

    def show_img(img):
        fig = plt.figure(figsize = (20,20))
        ax1 = fig.add_subplot(1,2,1)
        ax1.imshow(img, cmap = 'gray')
        ax1.set_title('image'), ax1.set_xticks([]), ax1.set_yticks([])

    # Parameters
    BLUR_SIGMA = 3
    GRADIENT_THRESHOLD = 40


    # Load image in different formats
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Blur the image
    img_blur = cv2.GaussianBlur(img_gray, (0,0), BLUR_SIGMA)

    # == Gradient computations == #

    # 1. Computer the sobel operator
    sobel_x = cv2.Sobel(img_blur, cv2.CV_64F,1,0)
    sobel_y = cv2.Sobel(img_blur, cv2.CV_64F,0,1)

    # 2. Compute gradient
    gradient_mag = np.sqrt(sobel_x * sobel_x + sobel_y * sobel_y)
    gradient_dir = cv2.phase(np.array(sobel_x, np.float32), np.array(sobel_y, dtype=np.float32), angleInDegrees=True)

    # 3. Compute magniture mask
    mask_mag = (gradient_mag > GRADIENT_THRESHOLD)
    #gradient_mag_corr = mask_mag * gradient_mag

    # 4. Compute pos/neg sobel masks
    mask_sobelx_pos = (sobel_x > 0)
    mask_sobelx_neg = (sobel_x < 0)
    mask_sobely_pos = (sobel_y > 0)
    mask_sobely_neg = (sobel_y < 0)

    # == Color based masks ==

    # 1. Compute white mask
    white_lower_hsv = np.array([0, 0, 150])
    white_upper_hsv = np.array([180, 50, 255])
    mask_white = cv2.inRange(img_hsv, white_lower_hsv, white_upper_hsv)

    # 2. Compute yellow mask
    yellow_lower_hsv = np.array([20, 100, 125])
    yellow_upper_hsv = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, yellow_lower_hsv, yellow_upper_hsv)

    # == left/right masks ==
    mask_left = np.ones(img_bgr.shape[:-1])
    mask_left[:,w//2:w + 1] = 0

    mask_right = np.ones(img_bgr.shape[:-1])
    mask_right[:,0:w//2] = 0

    #show_img(mask_mag )
    #print(mask_left.shape, mask_mag.shape, mask_sobelx_neg.shape, mask_sobely_neg.shape,)
    mask_left_edge = mask_left * mask_mag * mask_sobelx_neg * mask_sobely_neg * mask_yellow
    mask_right_edge = mask_right * mask_mag * mask_sobelx_pos * mask_sobely_neg * mask_white

    return mask_left_edge, mask_right_edge
