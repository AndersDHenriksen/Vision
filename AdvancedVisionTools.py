from pathlib import Path
import numpy as np
import cv2
try:
    from . import VisionTools as vt
    from . import FileTools as ft
except ImportError:
    import VisionTools as vt
    import FileTools as ft


def grabcut_using_mask(image, gc_mask, iterations=5, modify_mask_in_place=False):
    """
    Perform GrabCut on the image using the given GrabCut mask.
    - cv2.GC_BGD    (= 0) (Certain background)  := Not in any input mask
    - cv2.GC_FGD    (= 1) (Certain foreground)  := certain_fg
    - cv2.GC_PR_BGD (= 2) (Probably background) := potential_fg, not probable_fg and certain_fg
    - cv2.GC_PR_FGD (= 3) (Probably foreground) := probable_fg, not certain_fg

    :param image: Color image.
    :type image: np.core.multiarray.ndarray
    :param gc_mask: GrabCut mask.
    :type gc_mask: np.core.multiarray.ndarray
    :param iterations: Number of iterations that the GrabCut algorithm should run.
    :type iterations: int
    :param modify_mask_in_place: If True, the gc_mask will be modified in-place, so gc_mask_out == gc_mask.
    :type modify_mask_in_place: bool
    :return: (fg_mask, gc_mask_out) i.e. a binary foreground mask (certain + probable fg) and the updated GrabCut mask.
    :rtype: tuple(np.core.multiarray.ndarray)
    """
    assert image.shape[:2] == gc_mask.shape[:2], \
        "Dimension mismatch: {} vs. {}".format(image.shape[:2], gc_mask.shape[:2])
    assert gc_mask.dtype == np.uint8, "Invalid GrabCut mask image. Dtype is {}.".format(gc_mask.dtype)
    assert gc_mask.max() <= cv2.GC_PR_FGD, "Invalid GrabCut mask image. Max. value is {}.".format(gc_mask.max())
    mask = gc_mask if modify_mask_in_place else gc_mask.copy()
    _bg_model = np.zeros((1, 65), np.float64)  # Internal storage for the GrabCut algorithm
    _fg_model = np.zeros((1, 65), np.float64)  # Internal storage for the GrabCut algorithm
    cv2.grabCut(image, mask, None, _bg_model, _fg_model, iterCount=iterations, mode=cv2.GC_INIT_WITH_MASK)
    fg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    fg_mask[(mask == 1) | (mask == 3)] = 255
    # This appears to be faster than:
    # fg_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
    gc_mask_out = mask
    return fg_mask, gc_mask_out


def grabcut_mask_from_foreground_masks(potential_fg, probable_fg, certain_fg):
    """
    Create a GrabCut mask image (black/white uint8) from three given foreground mask images.
    The conversion to grabcut mask values is defined like this:
    - cv2.GC_BGD    (= 0) (Certain background)  := Not in any input mask
    - cv2.GC_FGD    (= 1) (Certain foreground)  := certain_fg
    - cv2.GC_PR_BGD (= 2) (Probably background) := potential_fg, not probable_fg and certain_fg
    - cv2.GC_PR_FGD (= 3) (Probably foreground) := probable_fg, not certain_fg

    :param potential_fg: Pixels that are potentially part of the foreground. May include probable and certain fg pixels.
    :type potential_fg: np.core.multiarray.ndarray
    :param probable_fg: Pixels that are probably part of the foreground. May include certain foreground pixels.
    :type probable_fg: np.core.multiarray.ndarray
    :param certain_fg: Pixels that are certainly part of the foreground.
    :type certain_fg: np.core.multiarray.ndarray
    :return: GrabCut mask image
    :rtype: np.core.multiarray.ndarray
    """
    assert potential_fg.dtype == probable_fg.dtype == certain_fg.dtype == np.uint8, "Masks must be uint8."
    assert potential_fg.shape == probable_fg.shape == certain_fg.shape, "Masks must be of similar shape."
    assert potential_fg.ndim == 2, "Masks must be two-dimensional."
    assert potential_fg.max() == 255, "Potential foreground mask must contain at least one white pixel."
    gc_mask = potential_fg // 255 * cv2.GC_PR_BGD
    gc_mask[probable_fg == 255] = cv2.GC_PR_FGD
    gc_mask[certain_fg == 255] = cv2.GC_FGD
    return gc_mask


def grabcut_mask_as_alpha_overlay(image, gc_mask, bg_col=None, pbg_col=(0, 0, 255),
                                  pfg_col=(255, 0, 0), fg_col=(255, 191, 0), alpha=0.3):
    """
    Visualize a GrabCut mask using alpha overlays on a color image.
    :param image: Color image.
    :type image: np.core.multiarray.ndarray
    :param gc_mask: GrabCut mask image, see cv2.grabCut().
    :type gc_mask: np.core.multiarray.ndarray
    :param bg_col: Color of pixels classified as background (cv2.GC_BGD).
    :type bg_col: tuple/None
    :param pbg_col: Color of pixels classified as probably background (cv2.GC_PR_BGD).
    :type pbg_col: tuple/None
    :param pfg_col: Color of pixels classified as probably foreground (cv2.GC_PR_FGD).
    :type pfg_col: tuple/None
    :param fg_col: Color of pixels classified as foreground (cv2.GC_FGD).
    :type fg_col: tuple/None
    :param alpha: Shared alpha value.
    :type alpha: float
    :return: Color image with alpha overlay.
    :rtype: np.core.multiarray.ndarray
    """
    image_gc_overlay = image
    if bg_col is not None:
        bg = (gc_mask == cv2.GC_PR_BGD).astype(np.uint8) * 255
        image_gc_overlay = vt.overlay_alpha_mask_on_image(image_gc_overlay, mask=bg, color=bg_col, alpha=alpha)
    if pbg_col is not None:
        pbg = (gc_mask == cv2.GC_PR_BGD).astype(np.uint8) * 255
        image_gc_overlay = vt.overlay_alpha_mask_on_image(image_gc_overlay, mask=pbg, color=pbg_col, alpha=alpha)
    if pfg_col is not None:
        pfg = (gc_mask == cv2.GC_PR_FGD).astype(np.uint8) * 255
        image_gc_overlay = vt.overlay_alpha_mask_on_image(image_gc_overlay, mask=pfg, color=pfg_col, alpha=alpha)
    if fg_col is not None:
        fg = (gc_mask == cv2.GC_FGD).astype(np.uint8) * 255
        image_gc_overlay = vt.overlay_alpha_mask_on_image(image_gc_overlay, mask=fg, color=fg_col, alpha=alpha)
    return image_gc_overlay


def draw_circle_on_image(image, center_uv, radius, color=(255, 0, 0), thickness=cv2.FILLED,
                         line_type=cv2.LINE_AA, inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    center_uv_int = tuple(np.round(center_uv).astype(np.int32))
    radius_int = np.round(radius).astype(np.int32)
    color = tuple(int(i) for i in color)
    cv2.circle(image_out, center_uv_int, radius_int, color=color, thickness=thickness, lineType=line_type)
    return image_out


def draw_line_segment_on_image(image, uv1, uv2, color=(255, 0, 0), thickness=1, line_type=cv2.LINE_AA, inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    pt1_int = tuple(np.round(uv1).astype(np.int32))
    pt2_int = tuple(np.round(uv2).astype(np.int32))
    color = tuple(int(i) for i in color)
    cv2.line(image_out, pt1_int, pt2_int, color=color, thickness=thickness, lineType=line_type)
    return image_out


def draw_rectangle_on_image(image, uv1, uv2, color=(255, 0, 0), thickness=1, line_type=cv2.LINE_AA, inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    pt1_int = tuple(np.round(uv1).astype(np.int32))
    pt2_int = tuple(np.round(uv2).astype(np.int32))
    color = tuple(int(i) for i in color)
    cv2.rectangle(image_out, pt1_int, pt2_int, color=color, thickness=thickness, lineType=line_type)
    return image_out


def draw_text_on_image(image, text, uv=(20, 50), font_face=cv2.FONT_HERSHEY_PLAIN, font_scale=3.0, color=(255, 0, 0),
                       thickness=3, line_type=cv2.LINE_AA, inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    pt_int = tuple(np.round(uv).astype(np.int32))
    color = tuple(int(i) for i in color)
    cv2.putText(image_out, text, pt_int, font_face, font_scale, color, thickness=thickness, lineType=line_type)
    return image_out


def draw_points_on_image(image, points_uv, point_size=15, color=(255, 0, 0), inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    points_uv = vt.intr(np.array(points_uv))
    skeleton_mask = np.zeros(image.shape[:2], bool)
    skeleton_mask[points_uv[:, 1], points_uv[:, 0]] = True
    skeleton_mask = vt.morph('dilate', skeleton_mask, (point_size, point_size), strel_kind='circle_big')
    image_out[skeleton_mask] = color
    return image_out


def draw_mask_on_image(image, mask, color=(255, 0, 0), dilate_r=0, inplace=False):
    image_out = ensure_image_for_drawing(image, color, inplace)
    if dilate_r:
        mask = vt.morph('dilate', mask, (dilate_r, dilate_r), 'circle_big')
    image_out[mask] = color
    return image_out


def ensure_image_for_drawing(image, color, inplace):
    converted = False
    if image.dtype == bool:
        image = 255 * image.astype(np.uint8)
        converted = True
    if image.ndim == 2 and np.ptp(color) > 0:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        converted = True
    if inplace and converted:
        print("Warning. Image had to converted, so inplace drawing not performed")
    return image if (inplace and not converted) else image.copy()


def circle_detection(image, radii, accum_thres, slack=0, center_outside=False, output='ijr'):
    """
    Find prominent circle  in bw image. Can either output prominent circle ijr coordinate or circle space which can then
    be used with scikit-image hough_circle_peaks. If there's no circle above accum_thres, output is None.
    :param image: Image to find circle edge in.
    :type image: np.core.multiarray.ndarray
    :param radii: Radii to search for. Only integers are supported.
    :type radii: Union[list, tuple, np.core.multiarray.ndarray]
    :param accum_thres: Threshold for circle support.
    :type accum_thres: float
    :param slack: Expansion width of circle threshold. Only integer is supported.
    :type slack: int
    :param center_outside: Whether to look for circles whose center is outside of image.
    :type center_outside: bool
    :param output: Whether to output circle ijr or hough-circle space. Must be either 'ijr' (default) or 'space'.
    :type output: str
    :return: Circle i, j, radius or transformed hough-circle space.
    :rtype: Union[NoneType, np.core.multiarray.ndarray]
    """
    tm_spaces = []
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    if center_outside:
        r = max(radii)
        image = cv2.copyMakeBorder(image, r, r, r, r, cv2.BORDER_CONSTANT, 0)
    for r in radii:
        circ_stel = vt.circular_structuring_element(2 * r + 1)
        circ_stel[1:-1, 1:-1] -= vt.circular_structuring_element(2 * r - 1)
        if slack:
            circ_stel = vt.morph('dilate', circ_stel, [2 * slack + 1] * 2)
        if circ_stel.shape[0] > image.shape[0] or circ_stel.shape[1] > image.shape[1]:
            break
        tm_out = cv2.matchTemplate(image, circ_stel, cv2.TM_CCORR_NORMED)
        tm_space = cv2.copyMakeBorder(tm_out, r, r, r, r, cv2.BORDER_CONSTANT, 0)
        tm_spaces.append(tm_space)
    circle_space = np.dstack(tm_spaces)
    if output == 'space':
        return circle_space
    circle_ijr = np.array(vt.argmax_nd(circle_space))
    accum = circle_space[circle_ijr[0], circle_ijr[1], circle_ijr[2]]
    circle_ijr[2] = radii[circle_ijr[2]]
    if slack:
        r = circle_ijr[2]
        circ_stel = vt.circular_structuring_element(2 * r + 1)
        circ_stel[1:-1, 1:-1] -= vt.circular_structuring_element(2 * r - 1)
        accum_thin = cv2.matchTemplate(vt.image_crop(image, circle_ijr[:2], [2 * r + 1] * 2, True), circ_stel, cv2.TM_CCORR_NORMED)
        accum = max(accum, accum_thin)
    if accum > accum_thres:
        if center_outside:
            circle_ijr[:2] -= max(radii)
        return circle_ijr


def minimum_path(begin_pt, end_pt, cost_matrix, step_size=1):
    """
    Find the path going from begin_pt to end_pt with minimum cost. The algorithm steps through the needed u-coordinates
    with a maximum change in v given by step_size. Thus, the algorithm works better if u distance >> v distance.
    :param begin_pt: uv starting point
    :type begin_pt: Union[list, tuple, np.core.multiarray.ndarray]
    :param end_pt: uv end point
    :type end_pt: Union[list, tuple, np.core.multiarray.ndarray]
    :param cost_matrix: cost landscape to find minimum path through
    :type cost_matrix: np.core.multiarray.ndarray
    :param step_size: maximum vertical step in path
    :type step_size: int
    :return: tuple of path u, v coordinates
    :rtype: tuple
    """
    if cost_matrix.dtype == np.uint8:   # To prevent overflow
        cost_matrix = cost_matrix.astype(int)
    inf = np.array([(cost_matrix.max() + 1) * cost_matrix.shape[1]], dtype=cost_matrix.dtype)
    infv = [np.tile(inf, i) for i in range(2 * step_size + 1)]
    inf0 = infv[step_size]
    inf2 = infv[2 * step_size]
    path_u = np.arange(begin_pt[0], end_pt[0] + 1)
    if len(path_u) <= begin_pt[1] - end_pt[1]:
        return None, None
    if len(path_u) == 1:
        return np.array([begin_pt[0]]), np.array([end_pt[1]])
    path_v = np.zeros_like(path_u)
    path_v[0], path_v[-1] = (begin_pt[1], end_pt[1])
    choices = []
    current_cost = np.array([0])
    v_positions = np.array([begin_pt[1]])
    upper_limit = cost_matrix.shape[0] - 1
    v_start = v_end = begin_pt[1]
    expand_lower_partial = expand_upper_partial = True
    for j in path_u[:-1]:
        expand_lower = expand_lower_partial and (v_positions[0] >= step_size)
        expand_upper = expand_upper_partial and (v_positions[-1] <= upper_limit - step_size)
        if expand_lower_partial and not expand_lower:
            current_cost = np.concatenate((infv[v_start], current_cost))
            v_start = 0
            expand_lower_partial = False
        if expand_upper_partial and not expand_upper:
            current_cost = np.concatenate((current_cost, infv[upper_limit - v_end]))
            v_end = upper_limit
            expand_upper_partial = False
        if expand_upper and expand_lower:
            v_start -= step_size
            v_end += step_size
            current_cost = np.concatenate((inf2, current_cost, inf2))
        elif expand_upper:
            v_end += step_size
            current_cost = np.concatenate((inf0, current_cost, inf2))
        elif expand_lower:
            v_start -= step_size
            current_cost = np.concatenate((inf2, current_cost, inf0))
        else:
            current_cost = np.concatenate((inf0, current_cost, inf0))
        if v_positions.size != v_end - v_start + 1:
            v_positions = np.arange(v_start, v_end + 1)
        cc = current_cost
        cc_rol = np.lib.stride_tricks.as_strided(cc, [cc.size - 2 * step_size, 1 + 2 * step_size], (cc.strides[0], cc.strides[0]))
        cc_argmin = cc_rol.argmin(axis=1)
        choices.append((v_positions[0], step_size - cc_argmin))
        current_cost = cost_matrix[v_positions, j + 1] + cc_rol[np.arange(cc_rol.shape[0]), cc_argmin]
    for j in range(2, len(path_u))[::-1]:
        path_v[j - 1] = path_v[j] - choices[j - 1][1][path_v[j] - choices[j - 1][0]]
    return path_u, path_v


def fit_line(x, y):
    """
    Equivalent to  numpy.polyfit, but using the faster cv2.fitLine backend
    :param x: 1D vector
    :type x: np.core.multiarray.ndarray
    :param y: 1D vector
    :type y: np.core.multiarray.ndarray
    :return:  Tuple of (slope, intercept)
    :rtype: np.core.multiarray.ndarray
    """
    params = cv2.fitLine(np.vstack((x, y)).T, cv2.DIST_L2, 0, 0.01, 0.01)

    # Start
    start_x = float(params[2] - params[0])
    start_y = float(params[3] - params[1])

    # End
    end_x = float(params[2] + params[0])
    end_y = float(params[3] + params[1])

    # Compute slope and intercept
    slope = (end_y - start_y) / (end_x - start_x)
    intercept = start_y - (start_x * slope)

    return np.array([slope, intercept])


def fit_circle(x, y):
    """
    Fit a circle to x, y points.
    :param x: 1D vector of x-coordinates
    :type x: np.core.multiarray.ndarray
    :param y: 1D vector of y-coordinates
    :type y: np.core.multiarray.ndarray
    :return: Tuple of (x_c, y_c, r)
    :rtype: tuple
    """
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    A = np.hstack((2 * x, 2 * y, np.ones_like(x)))
    b = x ** 2 + y ** 2
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    c = np.squeeze(c)
    x_c, y_c = c[0], c[1]
    r = np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
    return x_c, y_c, r


def save_video_from_folder(folder_name, pattern="*.png", frame_rate=20):
    """ Wrapper for save_video_clip that takes all images in a folder """
    save_video_clip("images", sorted(Path(folder_name).glob(pattern)), frame_rate)


def save_video_clip(file_name, frame_list, frame_rate=20):
    """
    Take a list of images and convert them to a video clip. Example usage:
    save_video_clip("time_lapse", list(Path(r"images").glob("*.png"))
    :param file_name: Output video file name. cwd / file_name .mp4
    :type file_name: str
    :param frame_list: List of images. Can also be list of image paths
    :type frame_list: Union[list, tuple, np.core.multiarray.ndarray]
    :param frame_rate: Video fps
    :type frame_rate: Union[int, float]
    :return: None
    :rtype: NoneType
    """
    # Misc frame list handling
    if not len(frame_list):
        print("Error empty framelist")
        return
    if isinstance(frame_list[0], str) or isinstance(frame_list[0], Path):
        frame_list = [cv2.imread(str(fp)) for fp in frame_list]
    frame_sizes = np.array([f.shape for f in frame_list])
    if np.ptp(frame_sizes, axis=0).any():
        print("Warning: Frames have different sizes will crop to same size")
        min_size = frame_sizes.min(axis=0)
        frame_list = [frame[:min_size[0], :min_size[1], :] for frame in frame_list]

    ft.download_h264_codec()
    codec = 'avc1'  # 'avc1'  # Alternative 'mp4v' 'xvid'
    clip_path = Path.cwd() / f"{file_name}.mp4"
    clip_path.parent.mkdir(exist_ok=True)

    print(f"Saving clip: ", end='')
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(clip_path), fourcc, frame_rate, frame_list[0].shape[:2][::-1])
    for frame in frame_list:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        out.write(frame)

    out.release()
    print(clip_path, flush=True)


def video_frame_generator(video_path):
    """
    Yield all frames from video file. First frame is used to determine if grayscale.
    :param video_path: Path to input video
    :param video_path: Union[str, path]
    :return: Consequetive frames
    :rtype: np.ndarray
    """
    is_gray = None
    cap = cv2.VideoCapture(str(video_path))
    video_frame_generator.fps = cap.get(cv2.CAP_PROP_FPS) or 20
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if is_gray is None:
            is_gray = np.ptp(frame, axis=-1).max() == 0
        if is_gray:
            yield frame[:, :, 0]
        else:
            yield frame
    cap.release()


def transcode_video(video_path):
    """
    Take a video file and convert it to h264 format.
    :param video_path: Path to input video
    :type video_path: Union[str, path]
    :return: None
    :rtype: NoneType
    """
    video_path = Path(video_path)
    new_path = video_path.parent / f"{video_path.stem}_h264"
    frame_list = list(video_frame_generator(video_path))
    save_video_clip(new_path, frame_list, video_frame_generator.fps)


def tesseract_ocr(image_bgr, is_digit=False):
    """
    Try to find text in image (ocr) using google tesseract.
    :param image_bgr: Input image to find text in
    :type image_bgr: np.ndarray
    :param is_digit: Whether input is a number
    :type is_digit: bool
    :return: Identified text string
    :rtype: str
    """
    import pytesseract  # conda install -c conda-forge pytesseract
    # Binary installer from: https://github.com/UB-Mannheim/tesseract/wiki
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
    return pytesseract.image_to_string(image_bgr, config='digits --psm 7' if is_digit else '--psm 7')  # Maybe add: --oem 3

def easy_ocr(image_bgr):
    """
    Try to find text in image (ocr) using easyocr.
    :param image_bgr: Input image to find text in
    :type image_bgr: np.ndarray
    :return: Identified text string
    :rtype: str
    """
    # From https://github.com/JaidedAI/EasyOCR
    import easyocr  # pip install easyocr
    if not hasattr(easy_ocr, 'reader'):
        easy_ocr.reader = easyocr.Reader(['en'])  # this needs to run only once to load the model into memory
    return easy_ocr.reader.readtext(image_bgr)


class OutlineComparer:
    def __init__(self, training_mask):
        """
        Create object for comparing blobs/connected components outlines, meaning if same geometrical outline, including
        rotation, then the outline distance will be small.
        :param training_mask: Binary mask with a single blob to use as the reference/training object.
        :type training_mask: np.ndarray
        """
        self.compare_angle = None  # This will hold the angle between reference and compared blobs
        self._centroid_vectors = None
        self.ref_curve = self._calc_dist_curve(training_mask)
        self._ref_angle = np.arctan2(*self._centroid_vectors[0])

    def outline_distance(self, query_mask):
        """
        Compare query blob to reference blob in terms of average contour distance. Further, an approximate angle
        difference is determined.
        :param query_mask: Binary mask with a single blob to compare against reference blob
        :type query_mask: np.ndarray
        :return: The average pixel distance between query blob and reference blob
        :rtype: float
        """
        input_curve = self._calc_dist_curve(query_mask)
        cost_landscape = np.abs(self.ref_curve[:, None] - input_curve[None, :])

        # Walk around both shapes in constant tempo
        n_eval = min(500, self.ref_curve.size, input_curve.size)
        ref_idx = vt.intr(np.linspace(0, self.ref_curve.size - 1, n_eval))  # cost_landscape: i
        new_idx = vt.intr(np.linspace(0, input_curve.size - 1, n_eval))  # cost_landscape: j
        new_offsets = vt.intr(np.linspace(0, input_curve.size - 1, 101))[:-1]
        new_idx_all = (new_idx[:, None] + new_offsets[None, :]) % input_curve.size
        mean_dist = cost_landscape[ref_idx[:, None], new_idx_all].mean(axis=0)

        self.compare_angle = self._ref_angle - np.arctan2(*self._centroid_vectors[new_offsets[mean_dist.argmin()]])
        return mean_dist.min()

    def _calc_dist_curve(self, mask):
        """
        Calculate the distance from mask centroid to each contour pixel.
        :param mask: Binary mask with a single blob to calculate distance curve for
        :type mask: np.ndarray
        :return: Distance curve.
        :rtype: np.ndarray
        """
        areas_num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8))
        assert areas_num == 2, "Multiple blobs in mask"
        # cv2.RETR_EXTERNAL # for only external contour. CHAIN_APPROX_NONE for all, but not starting from same point
        # RETR: https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
        contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = np.squeeze(np.vstack(contours))
        centroid_uv = vt.uv_centroid(mask)
        self._centroid_vectors = contour - centroid_uv
        return np.linalg.norm(self._centroid_vectors, axis=1)


if __name__ == "__main__":
    pass
