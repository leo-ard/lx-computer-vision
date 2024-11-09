from typing import Tuple

from matplotlib import pyplot as plt
from duckietown.utils.image.ros import compressed_imgmsg_to_rgb, rgb_to_compressed_imgmsg
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
import rospy


def clamp_matrix(m):
        max_value = np.max(m)
        clamping = (max_value // 3)
        temp = (m + clamping // 2) // clamping 
        return np.clip(temp, -1, 1)

def get_steer_matrix_from_points(x_1, y_1, x_2, y_2, size_x, size_y):
        m = (y_2 - y_1) / (x_2 - x_1)
        b = y_1 - m * x_1

        A = y_2 - y_1
        B = x_1 - x_2
        C = x_2 * y_1 - x_1 * y_2

        return np.array([[(A * i + B * j + C) for j in range(size_y)] for i in range(size_x)])

PUBLISHERS = {}
def publish_image(name, image, publisher = False):
    if publisher == False:
        if name in PUBLISHERS:
            publisher = PUBLISHERS[name]
        else:
            veh = rospy.get_namespace().strip("/")
            publisher = rospy.Publisher(f"/{veh}/debug/{name}/image/compressed", CompressedImage, queue_size=1)
            PUBLISHERS[name] = publisher

    compressed_image = rgb_to_compressed_imgmsg(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB), "jpeg")
    publisher.publish(compressed_image)


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

        left_steer_matrix = get_steer_matrix_from_points(x_1, y_1, x_2, y_2, size_x, size_y)
        left_mask = np.array([[1 if i > size_x // 3 and j < size_y // 2 else 0 for j in range(size_y)] for i in range(size_x)])

        left_matrix = -clamp_matrix(left_steer_matrix * left_mask)
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

        right_steer_matrix = get_steer_matrix_from_points(x_1, y_1, x_2, y_2, size_x, size_y)
        right_mask = np.array([[1 if i > size_x // 3 and j < size_y // 2 else 0 for j in range(size_y)] for i in range(size_x)])

        right_matrix = -clamp_matrix(right_steer_matrix * right_mask)
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
    BLUR_SIGMA = 4
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
    white_lower_hsv = np.array([45, 0, 175])
    white_upper_hsv = np.array([150, 100, 255])
    mask_white = cv2.inRange(img_hsv, white_lower_hsv, white_upper_hsv)
    #show_img(mask_white)


    # 2. Compute yellow mask
    yellow_lower_hsv = np.array([20, 40, 125])
    yellow_upper_hsv = np.array([40, 255, 255])
    mask_yellow = cv2.inRange(img_hsv, yellow_lower_hsv, yellow_upper_hsv)
    #show_img(mask_yellow)


    # == left/right masks ==
    mask_left = np.ones(img_bgr.shape[:-1])
    mask_left[:,w//2:w + 1] = 0

    mask_right = np.ones(img_bgr.shape[:-1])
    mask_right[:,0:w//2] = 0

    #show_img(mask_mag )
    #print(mask_left.shape, mask_mag.shape, mask_sobelx_neg.shape, mask_sobely_neg.shape,)
    #show_img(mask_sobelx_neg* mask_mag)
    #show_img(mask_sobely_neg * mask_mag)
    mask_grad_left = ((mask_sobelx_neg + mask_sobely_neg) / 2 ) * mask_mag
    mask_grad_right = ((mask_sobelx_pos + mask_sobely_neg) / 2 ) * mask_mag

    publish_image("sobelx_neg", mask_sobelx_neg * 255)
    publish_image("sobely_neg", mask_sobely_neg * 255)
    publish_image("sobelx_pos", mask_sobelx_pos * 255)
    publish_image("grad_left", mask_grad_left * 255)
    publish_image("grad_mag", mask_mag * 255)
    publish_image("grad_right", mask_grad_right * 255)
    publish_image("yellow", mask_yellow)
    publish_image("white",  mask_white)
    
    mask_left_edge = mask_left * mask_grad_left * mask_yellow
    mask_right_edge = mask_right * mask_grad_right * mask_white

    #show_img(mask_sobely_neg * mask_mag)
    #show_img(mask_sobelx_pos * mask_mag)
    #show_img(((mask_sobelx_neg + mask_sobely_neg) / 2) * mask_mag)


    return mask_left_edge, mask_right_edge
