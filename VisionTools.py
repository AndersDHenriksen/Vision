import numpy as np
import cv2


def morph(kind, image, strel_shape, strel_kind='rect'):
    """
    Wrapper for the different opencv morphology operations. Will also work on int/bool images and 1d vectors.
    :param kind: Which morpohological operation to perform. Possibilities are: 'erode', 'dilate', 'open', 'close',
    'gradient', 'tophat', 'whitehat', 'blackhat'.
    :type kind: str
    :param image: Source image to operate on.
    :type image: np.core.multiarray.ndarray
    :param strel_shape: Shape of kernel / structural element.
    :type strel_shape: Union[list, tuple, np.core.multiarray.ndarray, int]
    :param strel_kind: kernel / structural element type. Possibilities are: 'rect', 'cross', 'ellipse', 'circle_big',
    'circle_small'.
    :type strel_kind: str
    :return: Image after morphology operation.
    :rtype: np.core.multiarray.ndarray
    """
    assert kind in ['erode', 'dilate', 'open', 'close', 'gradient', 'tophat', 'whitehat', 'blackhat']
    assert strel_kind in ['rect', 'cross', 'ellipse', 'circle_big', 'circle_small']
    is_image_bool = image.dtype == np.bool
    is_image_int = image.dtype == np.int
    is_image_1d = len(image.shape) == 1
    if not isinstance(strel_shape, tuple):
        strel_shape = tuple(strel_shape if hasattr(strel_shape, '__iter__') else [strel_shape])

    if is_image_bool:
        image = image.astype(np.uint8)
    if is_image_int:
        image = image.astype(np.float)
    if is_image_1d:
        image = image[:, np.newaxis]
        if len(strel_shape) == 1:
            strel_shape = (1, strel_shape[0])

    if strel_kind in ['circle_big', 'circle_small']:
        kernel = circular_structuring_element(strel_shape[0], strel_kind == 'circle_big')
    else:
        strel_types = {'rect': cv2.MORPH_RECT, 'cross': cv2.MORPH_CROSS, 'ellipse': cv2.MORPH_ELLIPSE}
        kernel = cv2.getStructuringElement(strel_types[strel_kind], strel_shape)
    if kind == 'erode':
        out = cv2.erode(image, kernel)
    elif kind == 'dilate':
        out = cv2.dilate(image, kernel)
    else:
        op = {'open': cv2.MORPH_OPEN, 'close': cv2.MORPH_CLOSE, 'gradient': cv2.MORPH_GRADIENT,
              'tophat': cv2.MORPH_TOPHAT, 'whitehat': cv2.MORPH_TOPHAT, 'blackhat': cv2.MORPH_BLACKHAT}
        out = cv2.morphologyEx(image, op[kind], kernel)

    if is_image_bool:
        out = out.astype(np.bool)
    elif is_image_int:
        out = intr(out)
    if is_image_1d:
        out = out[:, 0]
    return out


def bw_distance_transform(mask, measure_white_region=True):
    """
    Wrapper for cv2.distanceTransform.
    :param mask: Input mask. Will be converted to uint8
    :type mask: np.core.multiarray.ndarray
    :param measure_white_region: Whether to measure white or black regions
    :type measure_white_region: bool
    :return: Distance image
    :rtype: np.core.multiarray.ndarray
    """
    if not measure_white_region:
        mask = ~mask
    out = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return out


def bw_edge(mask, include_at_border=False):
    """
    Find edges in binary mask. Edge pixels at the mask borders can optionally be included.
    :param mask: Mask to find edge in
    :type mask: np.core.multiarray.ndarray
    :param include_at_border: Whether to include edges at the border
    :type include_at_border: bool
    :return: Binary edge mask
    :rtype: np.core.multiarray.ndarray
    """
    mask = mask.astype(np.uint8)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    edge_mask = cv2.bitwise_xor(mask, cv2.erode(mask, kernel3))
    if include_at_border:
        edge_mask[:, [0, -1]] = mask[:, [0, -1]]
        edge_mask[[0, -1], :] = mask[[0, -1], :]
    return edge_mask.astype(np.bool)


def bw_edge_sorted(mask):
    """
    Get an array of edge points. The array is ordered so mask-neighbour points are array-neighbours.
    :param mask: Mask in which the largest edge will be sorted into array
    :type mask: np.core.multiarray.ndarray
    :return: Sorted edge points
    :rtype: np.core.multiarray.ndarray
    """

    # Preprocess mask
    mask2 = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), bool)
    mask2[1:-1, 1:-1] = mask
    mask2 = morph('erode', mask2, (3, 3))
    mask2 = bw_area_filter(mask2)
    mask2 = morph('dilate', mask2, (3, 3))
    mask_erode = morph('erode', mask2, (3, 3), 'cross')
    mask_dilate = morph('dilate', mask_erode, (3, 3), 'cross')
    mask_edge = mask_dilate ^ mask_erode

    # Walk around edge
    edge_points = [np.argwhere(mask_edge)[0]]
    ep = edge_points[0]
    mask_edge[ep[0], ep[1]] = False
    edge_crop = mask_edge[ep[0] - 1: ep[0] + 2, ep[1] - 1: ep[1] + 2]
    neighbour_points = np.argwhere(edge_crop)
    end_point = ep + neighbour_points[-1] - 1
    while edge_points[-1][0] != end_point[0] or edge_points[-1][1] != end_point[1]:
        ep = edge_points[-1]
        mask_edge[ep[0], ep[1]] = False
        edge_crop = mask_edge[ep[0] - 1: ep[0] + 2, ep[1] - 1: ep[1] + 2]
        neighbour_points = np.argwhere(edge_crop)
        edge_points.append(ep + neighbour_points[0] - 1)
    return np.array(edge_points) - 1


def bw_reconstruct(marker, mask):
    """
    Performs morphological reconstruction of the image marker under the image mask. Inspired by MATLABs bwreconstruct.
    [0, 0, 0]   [1, 1, 0]    [0, 0, 0]
    [0, 0, 0] , [0, 0, 0] -> [0, 0, 0]
    [0, 1, 0]   [1, 1, 1]    [1, 1, 1]
    :param marker: Binary mask indicating which of masks connected components to keep.
    :type marker: np.core.multiarray.ndarray
    :param mask: Binary mask with connected components which are filtered.
    :type mask: np.core.multiarray.ndarray
    :return: Filtered binary mask
    :rtype: np.core.multiarray.ndarray
    """
    areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    labels_to_keep = np.unique(labels[marker.astype(np.bool)])
    labels_to_keep = labels_to_keep[1:] if labels_to_keep.size and labels_to_keep[0] == 0 else labels_to_keep
    out_mask = np.isin(labels, labels_to_keep)
    return out_mask


def bw_area_filter(mask, n=1, area_range=(0, np.inf), output='mask'):
    """
    Filter objects from binary mask based on their size. Function can output both filtered mask and/or list of areas.
    :param mask: Binary mask to filter
    :type mask: np.core.multiarray.ndarray
    :param n: Number of largest connected components to keep for components inside area range
    :type n: int
    :param area_range: Range of acceptable areas to keep
    :type area_range: tuple
    :param output: String of what to output. Can either be 'both', 'area', or 'mask'.
    :param output: str
    :return: List of area or mask or tuple of both. All after filtering
    :rtype: Union[int, list, np.core.multiarray.ndarray]
    """
    assert mask.size, "Mask must have non-zero size"

    areas_num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
    areas = stats[:, cv2.CC_STAT_AREA][1:]
    inside_range_idx = np.flatnonzero((areas >= area_range[0]) & (areas <= area_range[1]))
    areas = areas[inside_range_idx]
    keep_idx = np.argsort(areas)[::-1][0:n]
    keep_areas = areas[keep_idx]
    if np.size(keep_areas) == 0:
        keep_areas = np.array([0])
    if n == 1:
        keep_areas = keep_areas[0]
    if output == 'area':
        return keep_areas
    keep_mask = np.isin(labels, inside_range_idx[keep_idx] + 1)
    if output == 'mask':
        return keep_mask
    return keep_mask, keep_areas


def bw_fill(mask):
    """
    Close interior holes in binary mask.
    :param mask: Binary mask to fill
    :type mask: np.core.multiarray.ndarray
    :return: Filled binary mask
    :rtype: np.core.multiarray.ndarray
    """
    mask_out = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv2.drawContours(mask_out, [cnt], 0, 255, -1)
    return mask_out.astype(np.bool)


def bw_convexhull(mask):
    """
    Get the convex hull mask of a binary mask, using opencv findContours.
    :param mask: Input binary mask
    :type mask: np.core.multiarray.ndarray
    :return: Convex hull of input mask
    :rtype mask: np.core.multiarray.ndarray
    """
    mask_out = mask.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # for each contour create convex hull object
    hull = []
    for contour in contours:
        hull.append(cv2.convexHull(contour, False))
    cv2.drawContours(mask_out, hull, contourIdx=-1, color=(255), thickness=-1)
    return mask_out.astype(np.bool)


def bw_clear_border(mask):
    """
    Remove any connected components touching the image border.
    :param mask: Binary mask to filter
    :type mask: np.core.multiarray.ndarray
    :return: Filtered binary mask
    :rtype: np.core.multiarray.ndarray
    """
    marker = np.zeros_like(mask)
    marker[(0, -1), :] = marker[:, (0, -1)] = True
    return mask ^ bw_reconstruct(marker, mask)


def bw_remove_empty_lines(image):
    good_rows = image.any(axis=1)
    image = image[good_rows, :]
    good_columns = image.any(axis=0)
    return image[:, good_columns]


def bw_left_and_right_edge(bw_image):
    """
    Find the left and right edge, i.e. left/right-most True pixels idx for each row. Speedup by first calling bw_edge.
    :param bw_image: Binary mask to find left and right edge in
    :type bw_image: np.ndarray
    :return: (left_edge, right_edge) as np.arrays
    :rtype: tuple
    """
    assert bw_image.size and bw_image.any(axis=1).all()
    us_vs = cv2.findNonZero(bw_image.astype(np.uint8))
    us, vs = us_vs[:, 0, 0], us_vs[:, 0, 1]
    right_edge = us[np.append(np.diff(vs) > 0, True)]
    left_edge = us[np.append(True, np.diff(vs) > 0)]
    return left_edge, right_edge


def bw_top_and_bottom_edge(bw_image):
    """ See bw_left_and_right_edge """
    return bw_left_and_right_edge(bw_image.T)


def imgradient(img):
    """
    Calculates the (Sobel) gradient magnitude of the image.
    :param img: Image from which to cacluate gradients
    :type img: np.core.multiarray.ndarray
    :return: Gradient magnitude image
    :rtype: np.core.multiarray.ndarray
    """
    return np.sqrt(cv2.Sobel(img, cv2.CV_64F, 1, 0) ** 2 + cv2.Sobel(img, cv2.CV_64F, 0, 1) ** 2)


def cc_masks(mask):
    """
    Extract each connected component (cc) in its own image and return the array of these images
    :param mask: Binary mask to split into array with each cc separated
    :type mask: np.core.multiarray.ndarray
    :return: Array of separate cc
    :rtype: np.core.multiarray.ndarray
    """
    areas_num, labels = cv2.connectedComponents(mask.astype(np.uint8))
    cc_stack = np.zeros((areas_num - 1, labels.shape[0], labels.shape[1]), np.bool)
    i_cor, j_cor = np.meshgrid(np.arange(labels.shape[0]), np.arange(labels.shape[1]), indexing='ij')
    mask = mask.astype(np.bool)
    cc_stack[labels[mask] - 1, i_cor[mask], j_cor[mask]] = True
    return cc_stack, labels


def argmin_nd(a):
    """
    Argmin on multidimensional array. E.g. for a 2D array the output is the (i, j) minimum index.
    :param a: Array to find minimum in
    :type a: np.core.multiarray.ndarray
    :return: Minimum (i, j, ...) index
    :rtype: tuple
    """
    return np.unravel_index(a.argmin(), a.shape)


def argmax_nd(a):
    """
    Argmax on multidimensional array. E.g. for a 2D array the output is the (i, j) maximum index.
    :param a: Array to find maximum in
    :type a: np.core.multiarray.ndarray
    :return: Maximum (i, j, ...) index
    :rtype: tuple
    """
    return np.unravel_index(a.argmax(), a.shape)


def uv_centroid(bw_image):
    M = cv2.moments(bw_image.astype(np.uint8))
    uv = intr(np.array([M["m10"], M["m01"]]) / M["m00"])
    return uv


def uv_coordinates(matrix, indexing='uv', center_origin=False):
    """
    Get coordinate matrixes with same size as input matrix. Coordinate matrix can either be uv or ij.
    :param matrix: Array whose shape will be used to determine coordinate matrix sizes.
    :type matrix: np.core.multiarray.ndarray
    :param indexing: String or either 'uv' or 'ij'
    :type indexing: str
    :param center_origin: Whether to center the origo in matrix/image center.
    :type center_origin: bool
    :return: Tuple of index matrices
    :rtype: tuple
    """
    v, u = np.arange(matrix.shape[0]), np.arange(matrix.shape[1])
    if center_origin:
        v, u = v - v[-1] / 2, u - u[-1] / 2
    return np.meshgrid(v, u, indexing='ij') if indexing == 'ij' else np.meshgrid(u, v)


def r_coordinates(matrix, unit_scale=False, also_return_angle=False):
    """
    Get radius matrix with same size as input matrix.
    :param matrix: Array whose shape will be used to determine radius matrix sizes.
    :type matrix: np.core.multiarray.ndarray
    :param unit_scale: Whether to scale the output so both axes go from -1 to 1
    :type unit_scale: bool
    :param also_return_angle: Whether to also return angle matrix. Aligned with uv coordinate system.
    :type also_return_angle: bool
    :return: Radius matrix
    :rtype: np.core.multiarray.ndarray
    """
    u, v = uv_coordinates(matrix)
    u_c, v_c = u - (matrix.shape[1] - 1)/2, v - (matrix.shape[0] - 1)/2
    if unit_scale:
        u_c, v_c = 2 * u_c / (matrix.shape[1] - 1), 2 * v_c / (matrix.shape[0] - 1)
    if also_return_angle:
        return np.sqrt(u_c ** 2 + v_c ** 2), np.arctan2(v_c, u_c)
    return np.sqrt(u_c ** 2 + v_c ** 2)


def simple_rotate(image, angle, out='rot_image'):
    """
    Rotate image without cropping it. Rotation is clockwise. simple_rotate can output multitple intermediates so
    "out" can be rot_matrix, rot_function, rot_image or tuple/list of these.
    :param image: Image to rotate
    :type image: np.core.multiarray.ndarray
    :param angle: Angle to rotate
    :type angle: float
    :param out: List (of strings) or string of what to output. Possibilities are:
                rot_matrix, invert_matrix, rot_function, rot_image.
    :type out: Union[list, str]
    :return: Rotated image, roated matrix or rotate function based on "out"
    :rtype: Union[np.core.multiarray.ndarray, list, function]
    """
    out = [out] if isinstance(out, str) else out
    output = []
    # grab the dimensions of the image and then determine the center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    rotate_matrix = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(rotate_matrix[0, 0])
    sin = np.abs(rotate_matrix[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    rotate_matrix[0, 2] += (nW / 2) - cX
    rotate_matrix[1, 2] += (nH / 2) - cY
    if 'rot_matrix' in out:
        output.append(rotate_matrix)
    if 'invert_matrix' in out:
        output.append(cv2.invertAffineTransform(rotate_matrix))
    rotate_function = lambda img: cv2.warpAffine(img.astype(np.uint8), rotate_matrix, (nW, nH)).astype(img.dtype)
    if 'rot_function' in out:
        output.append(rotate_function)
    if 'rot_image' in out:
        output.append(rotate_function(image))
    return output if len(output) > 1 else output[0]


def image_crop(image, point, sides, is_point_center=False, indexing='ij', assign=None):
    """
    Cropped an image based on side length and either midpoint or lower-left-corner point. Possibly assign to this sub
    image instead of returning the crop.
    :param image: Image to crop
    :type image: np.core.multiarray.ndarray
    :param point: Anchor point for crop. Point can either be lower-left-corner (i.e. lowest index each side) (default)
    or mid point of crop.
    :type point: Union[list, tuple, np.core.multiarray.ndarray]
    :param sides: Side length of crop. Sides can be negative. Meaning anchor is now lower-left-corner.
    :type sides: Union[list, tuple, np.core.multiarray.ndarray]
    :param is_point_center: Whether anchor point is crop center or lower-left-corner (default)
    :type is_point_center: bool
    :param indexing: Whether to use 'ij' (default) or 'uv' indexing.
    :type indexing: str
    :param assign: If this is not None, this will be assigned to the image crop. In this case the function returns None.
    :type assign: Union[NoneType, np.core.multiarray.ndarray]
    :return: Crop of image.
    :rtype: Union[NoneType, np.core.multiarray.ndarray]
    """
    assert len(point) == len(sides) == 2 and image.ndim >= 2 and indexing in ['ij', 'uv']

    point, sides = np.array(point), np.array(sides)
    if is_point_center:
        point[0] = max(0, point[0] - sides[0] // 2)
        point[1] = max(0, point[1] - sides[1] // 2)
    point, sides = intr(point), intr(sides)

    i = [point[0], point[0] + sides[0]] if sides[0] >= 0 else [max(0, point[0] + sides[0] + 1), point[0] + 1]
    j = [point[1], point[1] + sides[1]] if sides[1] >= 0 else [max(0, point[1] + sides[1] + 1), point[1] + 1]

    if indexing == 'uv':
        i, j = j, i

    if assign is None:
        return image[i[0]: i[1], j[0]: j[1], ...]
    image[i[0]: i[1], j[0]: j[1], ...] = assign


def circular_structuring_element(kernel_size, go_big=False):
    # Create a _symmetrical_ circular structuring element (unlike OpenCV's built-in circular structuring element)
    assert (kernel_size % 2 == 1)  # Only for odd kernel sizes
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    kernel2 = kernel1.transpose()
    if go_big:
        kernel = cv2.bitwise_or(kernel1, kernel2)  # Big kernel
    else:
        kernel = cv2.bitwise_and(kernel1, kernel2)  # Small kernel
    return kernel


def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and grab the image size
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), int(height))

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the dimensions
        r = width / float(w)
        dim = (int(width), int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def adjust_gamma(image, gamma=None, desired_intensity=None):
    assert gamma or desired_intensity, "gamma or desired_intensity must be set"
    if desired_intensity is not None:
        gamma = np.log(image.mean()/255) / np.log(desired_intensity/255)
    # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    table = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
    return cv2.LUT(image, table)


def find(a, if_none=np.nan):
    """
    Similar to np.flatnonzero but if no True elements in a, then find return np.array([if_none]) so find(a)[0] still
    works.
    :param a: Array from which True indexes are return
    :type a: np.core.multiarray.ndarray
    :param if_none: Element or fake-index to return if no True elements in a
    :type if_none: float
    :return: Array with indexes
    :rtype: np.core.multiarray.ndarray
    """
    a_idx = np.flatnonzero(a)
    return a_idx if a_idx.size else np.array([if_none])


def rolling_window(a, window):
    """
    Split array into rolling window components, e.g. [3,4,5] , window=2 -> [[3,4], [4,5]]
    :param a: Array to split into window size
    :type a: np.core.multiarray.ndarray
    :param window: Window size
    :type window: int
    :return: Array of window arrays
    :rtype: np.core.multiarray.ndarray
    """
    if a.size < window:
        return np.array([])
    return np.lib.stride_tricks.as_strided(a, [a.size + 1 - window, window], (a.strides[0], a.strides[0]))


def running_mean(x, N):
    """
    Calculate the running mean of an array.
    :param x: Array to calculate the running mean of.
    :type x: np.core.multiarray.ndarray
    :param N: Window size
    :type N: int
    :return: running mean values
    :rtype: np.core.multiarray.ndarray
    """
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def diff_long(x, stepsize):
    """
    Diff with a larger window. diff_long(x,1) = np.diff(x)
    :param x: Array to differentiate
    :type x: np.core.multiarray.ndarray
    :param stepsize:
    :type stepsize: int
    :return: diffed values
    :rtype: np.core.multiarray.ndarray
    """
    return x[stepsize:] - x[:-stepsize]


def find_clusters(a, allowed_jump=0, min_size=1, only_return_longest=False):
    """
    Find clusters, where the index has jumps/discontinuities, i.e. [1, 2, 3, 7, 8, 9] contains 2 clusters. Input can
    also be boolean array.

    :param a: Array for cluster search. Can be array of index or bool values.
    :type a: np.core.multiarray.ndarray
    :param allowed_jump: Discontinuities which should not be considered a cluster break
    :type allowed_jump: int
    :param min_size: Minimum cluster size to keep
    :type min_size: int
    :param only_return_longest: If true, will only return longest cluster
    :type only_return_longest: bool
    :return: List of clusters as np.arrays or longest cluster
    :rtype: Union[list, np.core.multiarray.ndarray]
    """

    # Convert to index if bool
    if a.dtype == bool:
        a = np.flatnonzero(a)

    # Walk through array and find clusters
    da = np.diff(a)
    clusters = []
    current_cluster_size = 1
    for i in range(1, len(a)):
        if da[i-1] > allowed_jump + 1:
            if current_cluster_size >= min_size:
                clusters.append(a[i-current_cluster_size:i])
            current_cluster_size = 1
        else:
            current_cluster_size += 1
    if current_cluster_size >= min_size and len(a):
        clusters.append(a[len(a) - current_cluster_size:])
    if only_return_longest and len(clusters):
        return sorted(clusters, key=len)[-1]
    return clusters


def curve_step(curve, begin_level, end_level):
    """
    Find the best index to split a curve into a step function based on lowest RMSE.
    :param curve: Curve to look for step in
    :type curve: np.core.multiarray.ndarray
    :param begin_level: Begin value of step function to fit
    :type begin_level: float
    :param end_level: End value of step function to fit
    :type end_level: float
    :return: Curve index where step function goes from begin_level to end_level
    :rtype: int
    """
    cost = np.array([np.sum((curve[:i] - begin_level)**2) + np.sum((curve[i:] - end_level)**2) for i in range(curve.size)])
    return cost.argmin()


def local_maximum(a, include_edge_maxima=False):
    """
    Get position of all local maximum in array.
    :param a: Array to find lccal maximum in
    :type a: np.core.multiarray.ndarray
    :param include_edge_maxima: Whether to include local maximum at ends of array
    :type include_edge_maxima: bool
    :return: Array of positions/index which are local maximum
    :rtype: np.core.multiarray.ndarray
    """
    if a is None or len(a) == 0:
        return np.array([])
    da = np.diff(a)
    dap = np.flatnonzero(da >= 0)
    dan = np.flatnonzero(da <= 0)
    not_plateau = da != 0
    if not dan.size or not dap.size:
        if not include_edge_maxima:
            return np.array([])
        return np.array(np.flatnonzero(a == a.max()).mean())
    dap_stops = dap[np.flatnonzero(np.append(np.diff(dap) > 1, True))] + 1
    if not include_edge_maxima and dap_stops[-1] == da.size:
        dap_stops = dap_stops[:-1]
    if not dap_stops.size:
        return np.array([])
    dan_stops = dan[np.flatnonzero(np.append(np.diff(dan) > 1, True))] + 1
    next_idx = 0 if include_edge_maxima or da[0] > 0 else dan_stops[0]
    is_dap_local_max = np.zeros(dap_stops.shape, np.bool)
    while next_idx < dap_stops[-1]:
        is_dap_local_max[find(dap_stops > next_idx)[0]] = True
        next_idx = dan_stops[dan_stops > next_idx][0] if dan_stops[-1] > next_idx else da.size
    local_max = dap_stops[is_dap_local_max]
    adjusted_local_max = np.array([lm - find(not_plateau[lm - 1::-1], if_none=lm)[0]/2 for lm in local_max])
    if include_edge_maxima and da[0] < 0:
        adjusted_local_max = np.insert(adjusted_local_max, 0, 0)
    return adjusted_local_max


def weighted_3point_extrema(array):
    """
    Find the (weighted) extrema index of 3 points, assuming middle point is the maximum or minimum of the 3 points.
    :param array: Array of 3 points.
    :type array: np.core.multiarray.ndarray
    :return: Sub-integer index of extrema.
    :rtype: float
    """
    assert array.size == 3
    p_coef = np.polyfit(np.arange(3), array, 2)
    return -p_coef[1] / p_coef[0] / 2


def sub_integer_extrema(array, extrema_index):
    """
    Take an extrema index of an array and find the sub-integer extrema index.
    :param array: Data array.
    :type array: np.core.multiarray.ndarray
    :param extrema_index: Index of an extrema in array.
    :type array: int
    :return: Sub-integer index of extrema.
    :rtype: float
    """
    return weighted_3point_extrema(array[extrema_index - 1: extrema_index + 2]) + extrema_index - 1


def direction_from_blob(blob_mask):
    """
    Gives the orientation/principal-component angle and centroid of mask.
    To rotate use: rot = vt.simple_rotate(blob_mask, -angle)
    :param blob_mask: Single component binary mask
    :type blob_mask: np.core.multiarray.ndarray
    :return: (angle_uv, centroid_uv)
    :rtype: tuple
    """
    moments = cv2.moments(blob_mask.astype(np.uint8), True)
    centroid_uv = intr(np.array([moments["m10"], moments["m01"]]) / moments["m00"])
    angle_uv = np.rad2deg(np.arctan2(moments['mu11'], (moments['mu20'] - moments['mu02']) / 2) / 2)
    return angle_uv, centroid_uv


def distance_point_to_line(line_pt1, line_pt2, pt):
    """
    Calculates the distance from point pt to the line going through line_pt1 and line_pt2.
    Note that the point can be either 2D (e.g. uv) or 3D (e.g. xyz).
    https://stackoverflow.com/a/39840218/1447415
    https://math.stackexchange.com/questions/1292212/cross-product-of-vectors-distance-from-a-line
    :param line_pt1: First point on the line given as an np.array of size 1x2 or 1x3.
    :type line_pt1: np.core.multiarray.ndarray
    :param line_pt2: Second point on the line given as an np.array of size 1x2 or 1x3.
    :type line_pt2: np.core.multiarray.ndarray
    :param pt: Point not (necessarily) on the line given as an np.array of size 1x2 or 1x3.
    :type pt: np.core.multiarray.ndarray
    :return: Distance from the point 'pt' to the line.
    :rtype: float
    """
    line_vec = line_pt2 - line_pt1
    dist = np.linalg.norm(np.cross(line_vec, line_pt1 - pt)) / np.linalg.norm(line_vec)
    return dist


def distance_to_fit(y):
    """
    Calcualte distance to best line fit.
    :param y: y-values to fit
    :type y: np.core.multiarray.ndarray
    :return: distance to fitted line
    :rtype: np.core.multiarray.ndarray
    """
    from .AdvancedVisionTools import fit_line
    x = np.arange(y.size)
    return y - np.polyval(fit_line(x, y), x)


def unit_vector(vector):
    """ Returns the unit vector of the vector. """
    return np.array(vector) / np.linalg.norm(vector)


def intr(a):
    """
    Round and convert to integer. Especially useful for indexing.
    :param a: Array or float to round and cast
    :type a: Union[float, list, tuple, np.core.multiarray.ndarray]
    :return: Rounded integer or array of ints
    :rtype: Union[float, list, tuple, np.core.multiarray.ndarray]
    """
    if isinstance(a, np.ndarray):
        return np.round(a).astype(np.int)
    if isinstance(a, list):
        return [int(round(e)) for e in a]
    if isinstance(a, tuple):
        return tuple(int(round(e)) for e in a)
    return int(round(a))


def confusion_matrix(y_label, y_predict, label_for_print=None):
    """
    Calculate and maybe plot confusion matrix. y_label and y_predict are assumed to increasing integers.
    :param y_label: True/Annotated labels
    :type y_label: np.core.multiarray.ndarray
    :param y_predict: Predicted/inferred labels
    :type y_predict: np.core.multiarray.ndarray
    :param label_for_print: Label array. If this variable is not none, the confusion matrix will be printed.
    :type label_for_print: list
    :return: Confusion matrix as nparray
    :rtype: np.core.multiarray.ndarray
    """
    y_label, y_predict = np.array(y_label).astype(np.int), np.array(y_predict).astype(np.int)
    n_max = max(y_label.max(), y_predict.max()) + 1
    cm = np.zeros((n_max, n_max), dtype=np.int)
    np.add.at(cm, (y_label, y_predict), 1)
    if label_for_print is not None:
        n_label = max(len(l) for l in label_for_print)
        n_number = max(len(str(i)) for i in cm.ravel())
        for label, row in zip(label_for_print, cm):
            print(f'{label:{n_label}} | ', end='')
            for i in row:
                print(f'{i:{n_number}} | ', end='')
            print('')
        print('')
        error = y_label != y_predict
        print(f'Error-rate: {np.sum(error)} / {error.size}')
        for i, label in enumerate(label_for_print):
            print(f"{label:{n_label}}: {np.sum(error[y_label==i])} / {np.sum(y_label==i)} | errors: {find(error & (y_label==i))}")

    return cm


def put_text(image, text, position_uv, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, color=(0, 0, 255), thickness=2):
    """ Wrapper for cv2.putText with more defaults. """
    if image.ndim == 2 and len(color) > 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    position_uv = tuple(intr(position_uv))
    return cv2.putText(image, text, position_uv, font, font_scale, color, thickness, cv2.LINE_AA)


def rgb_like(image, color="red"):
    """
    Create a single color image with the same size as the input image.
    :param image: Image for which the output will have the same size
    :type image: np.core.multiarray.ndarray
    :param color: The output color
    :type color: str
    :return: (H,W,3) single color uint8 image
    :rtype: np.core.multiarray.ndarray
    """
    color = color.lower()
    assert color in ["red", "green", "blue", "yellow", "magenta", "cyan"]
    image = np.squeeze(image)
    assert image.ndim in [2, 3]
    zeros = np.zeros(image.shape[:2], np.uint8)
    ones = 255 * np.ones(image.shape[:2], np.uint8)
    if color == "red":
        return np.dstack((ones, zeros, zeros))
    elif color == "green":
        return np.dstack((zeros, ones, zeros))
    elif color == "blue":
        return np.dstack((zeros, zeros, ones))
    elif color == "yellow":
        return np.dstack((ones, ones, zeros))
    elif color == "magenta":
        return np.dstack((ones, zeros, ones))
    elif color == "cyan":
        return np.dstack((zeros, ones, ones))


def add_overlay_to_image(image, mask, alpha=0.5, color="red", filename=None):
    """
    Adds a transparent, colored overlay to an image below a given mask.
    :param image: Image to add overlay to (not in-place). Can be monochrome or RGB.
    :type image: np.core.multiarray.ndarray
    :param mask: Greyscale mask. Values should be between 0 and 1.
    :type mask: np.core.multiarray.ndarray
    :param alpha: The opacity of the overlay. 0.0 means fully transparent, 1.0 means fully opaque.
    :type alpha: float
    :param color: The color of the overlay. Possible: red, green, blue, yellow, magenta, cyan
    :type color: str
    :param filename: Optional filename to save the overlayed images
    :type filename: str
    :return: A copy of the image with a transparent, colored overlay below the mask.
    :rtype: np.core.multiarray.ndarray
    """
    assert mask.dtype is not np.uint8
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))
    color_overlay = rgb_like(mask, color)
    alpha_mask = (alpha * mask).clip(min=0, max=1)[:, :, None]
    overlay_image = alpha_mask * color_overlay + (1 - alpha_mask) * image
    overlay_image = np.round(overlay_image).astype(np.uint8)
    if filename is not None:
        cv2.imwrite(filename, overlay_image)
    return overlay_image


def showimg(img, overlay_mask=None, close_on_click=False, title=None, cmap="gray", overlay_cmap="RdBu"):
    """
    Plot an RGB or grayscale image using matplotlib.pyplot

    :param img: Image to plot.
    :type img: np.core.multiarray.ndarray
    :param overlay_mask: Binary mask for colored overlay
    :type overlay_mask: np.core.multiarray.ndarray
    :param close_on_click: Whether to close figure on button press
    :type close_on_click: bool
    :param title: Title string
    :type title: str
    :param cmap: Colormap to use. Examples: gray, hot, hot_r, jet, jet_r, summer, rainbow, ...
    :type cmap: str
    :param overlay_cmap: Colormap for image. Examples: gray, hot, hot_r, jet, jet_r, summer, rainbow, ...
    :type overlay_cmap: str
    :return: Figure containing the plotted image
    :rtype: matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    # Figure/window size if not maximized
    figure_width_inches = 18.5
    figure_height_inches = 10.5

    # Window size
    fig = plt.figure()
    fig.set_size_inches(figure_width_inches, figure_height_inches, forward=True)

    # Show image
    img = np.squeeze(img)
    if img.ndim == 1:
        plt.plot(img)
    else:
        plt.imshow(img, cmap=cmap)

        if overlay_mask is not None:
            masked = np.ma.masked_where(overlay_mask == 0, overlay_mask)
            plt.imshow(masked, overlay_cmap, alpha=0.5)
    if title:
        plt.title(title)

    # Trim margins
    plt.tight_layout()  # plt.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
    if close_on_click:
        plt.waitforbuttonpress()
        plt.close(fig)
    else:
        plt.show()
    return fig


if __name__ == "__main__":
    pass
