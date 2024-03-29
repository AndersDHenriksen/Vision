import time
import json
import traceback
import configparser
from pathlib import Path
import numpy as np
import cv2
from pypylon import pylon, genicam
try:
    from . import VisionTools as vt
except ImportError:
    import VisionTools as vt


def get_camera_names():
    return [c.GetModelName() for c in pylon.TlFactory.GetInstance().EnumerateDevices()]


# class ImageHandler(pylon.ImageEventHandler):
#     def __init__(self, logger, camera_name='camera'):
#         super().__init__()
#         self.logger = logger
#         self.camera_name = camera_name
#         # Communication to main thread should be done with Queue, i.e. from queue import Queue
#
#     def OnImageGrabbed(self, camera, grab_result):
#         # Try/except needed as code is run in a pylon thread context, where errors can't be properly reported from the
#         # background thread to the foreground thread
#         try:
#             if not grab_result.GrabSucceeded():
#                 self.logger.warning(f"Grab Failed for {self.camera_name}")
#                 return
#
#             self.logger.info(f"Grab succeeded for {self.camera_name}")
#             image = grab_result.Array
#             # Do something with image
#
#         except Exception as e:
#             self.logger.error("\n".join(traceback.format_tb(e.__traceback__) + [str(e)]))
#         finally:
#             grab_result.Release()


class CameraWrapper:

    def __init__(self, exposure_time_us=None, trigger_method='software', enable_jumbo_frame=False):
        assert trigger_method in ['software', 'hardware', 'streaming']

        self.software_trigger = None
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self.camera.UserSetSelector.Value = "Default"
        self.camera.UserSetLoad.Execute()
        if enable_jumbo_frame and self.camera.IsGigE():
            self.camera.GevSCPSPacketSize.Value = 8192
            # self.camera.GevSCPD.Value = 1000
            # self.camera.GevSCFTD.Value = 1000
        if exposure_time_us is not None:
            self.camera.ExposureAuto.Value = 'Off'
            try:
                self.camera.ExposureTimeAbs.Value = exposure_time_us
            except genicam.LogicalErrorException:
                self.camera.ExposureTime.Value = exposure_time_us
        if trigger_method == 'streaming':
            self.setup_for_streaming()
        elif trigger_method == 'hardware':
            self.setup_for_hardware_trigger()
        else:
            self.setup_for_software_trigger()
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)  # pylon.GrabStrategy_OneByOne
        # Alternatively, start grabbing but handle incoming images in camera thread
        # self.handler = ImageHandler(...)  # Local reference to prevent garbage collection
        # self.camera.RegisterImageEventHandler(self.handler, pylon.RegistrationMode_ReplaceAll, pylon.Cleanup_Delete)
        # self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly, pylon.GrabLoop_ProvidedByInstantCamera)
        self.converter = None
        if self.camera.PixelFormat.Value == 'BayerRG8':
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed

    def setup_for_software_trigger(self):
        self.camera.TriggerMode.Value = 'On'
        self.camera.TriggerSource.Value = 'Software'
        self.camera.RegisterConfiguration(pylon.SoftwareTriggerConfiguration(), pylon.RegistrationMode_ReplaceAll,
                                          pylon.Cleanup_Delete)
        self.software_trigger = True

    def setup_for_hardware_trigger(self):
        self.camera.TriggerMode.Value = "On"
        self.camera.TriggerSource.Value = "Line1"
        self.software_trigger = False

    def setup_for_streaming(self):
        self.camera.TriggerMode.Value = "Off"
        self.software_trigger = False

    def grab(self, wait_for_image=True):
        if self.software_trigger:
            self.camera.ExecuteSoftwareTrigger()
        if not wait_for_image and not self.camera.GetGrabResultWaitObject().Wait(0):
            return
        image = None
        with self.camera.RetrieveResult(1000) as grabResult:
            if grabResult.GrabSucceeded():
                if self.converter is not None:
                    image = self.converter.Convert(grabResult).Array
                else:
                    image = grabResult.Array
        return image if image is not None else self.grab()  # Retry if grab failed

    def grab_single(self):
        return self.camera.GrabOne(1000).Array

    def stop(self):
        self.camera.StopGrabbing()
        self.camera.Close()

    def live_view(self, save_folder=Path.cwd()):
        print("Starting liveview. Controls are:\nspace: pause\nenter: save\nq\esc: quit")
        pause = False
        while True:
            if not pause:
                image = self.grab()
            if image.ndim == 3:
                image = image[:, :, ::-1]
            image_resized = cv2.resize(image, (1200, int(image.shape[0] / image.shape[1] * 1200)))
            cv2.imshow("Live View", image_resized)
            key = cv2.waitKey(10)
            if key in [ord('\x1b'), ord('q')]:
                break
            if key == ord(' '):
                pause = not pause
            if key == ord('\r'):
                filename = time.strftime(f"%Y-%m-%d %H-%M-%S", time.localtime()) + ".png"
                print(f"Saving image to {str(Path(save_folder) / filename)}")
                cv2.imwrite(str(Path(save_folder) / filename), image)


class CheckerboardCalibrator:
    def __init__(self, config_load_path=None):
        self.config = configparser.ConfigParser()
        self.CameraMatrix = np.eye(3)
        self.DistortionCoefficients = np.zeros((5,))
        self.PerspectiveTransform = None
        self.PixelsPerMm = None
        if config_load_path is not None:
            self.import_from_config_file(config_load_path)

    def calibrate_from_checkerboard(self, image, do_perspective_transform=False, square_side_length=None):
        """Calibration based on a checkerboard target. Assume black square on white background"""
        h, w = image.shape

        # Estimate angel
        grad_x, grad_y = cv2.Sobel(image, cv2.CV_64F, 1, 0), cv2.Sobel(image, cv2.CV_64F, 0, 1)
        grad_n, grad_a = np.sqrt(grad_x ** 2 + grad_y ** 2), np.arctan2(grad_x, grad_y)
        hist_count, hist_axis = np.histogram(grad_a, bins=360, weights=grad_n)
        rough_angle = hist_axis[hist_count.argmax()] * 180 / np.pi
        rough_angle = (rough_angle + 45) % 90 - 45  # keep it within -45, 45. Maybe not important
        img_rot = vt.simple_rotate(255 - image, rough_angle)  # Also invert, so white background is black like border
        # Count squares
        black_squares = vt.morph('erode', img_rot > 128, (11, 11))
        retval, labels, stats, centroids = cv2.connectedComponentsWithStats(black_squares.astype(np.uint8))
        areas = stats[:, 4]
        centroids = centroids[np.abs(1 - areas / np.median(areas)) < .2]
        d_centroid = np.diff(centroids[:, 1])
        cy = sum(d_centroid > d_centroid.max() / 2)
        cx = vt.intr(2 * centroids.shape[0] / (cy + 1) - 1)
        if abs(2 * centroids.shape[0] - (cx + 1) * (cy + 1)) > 1:
            print("Calibration Warning: Number of identified black squares is off.")

        # Finding the checkerboard corners
        corners_found, corners = cv2.findChessboardCorners(image, (cx, cy), cv2.CALIB_CB_ADAPTIVE_THRESH)
        if not corners_found:
            raise Exception("Calibration failed.")
        # vt.showimg(avt.draw_points_on_image(image, np.squeeze(corners)))  # import AdvancedVisionTools as avt

        # Refine corners. This doesn't do much actually
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.01)
        corners = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
        corners_grid = corners.reshape((cy, cx, 2))
        corners = np.squeeze(corners)

        # Make sure corners are in correct order
        d_corner = corners[1] - corners[0]
        if d_corner[0] < d_corner[1]:
            corners_grid = np.transpose(corners_grid, (1, 0, 2))[:, ::-1, :]
            corners = corners_grid.reshape(corners.shape)

        # Calculate camera calibration
        count_xy = np.arange(cx * cy)
        object_points = np.dstack((count_xy % cx, count_xy // cx, np.zeros_like(count_xy))).astype(np.float32)
        ret, self.CameraMatrix, self.DistortionCoefficients, rvecs, tvecs = cv2.calibrateCamera(object_points, corners[None, ...], (w, h), None, None)
        corners = cv2.undistortPoints(corners, self.CameraMatrix, self.DistortionCoefficients, P=self.CameraMatrix)[:, 0, :]

        if do_perspective_transform:
            img_4corners = np.vstack((corners[0], corners[cx - 1], corners[-cx], corners[-1]))
            calib_w = np.linalg.norm(corners_grid[:, -1] - corners_grid[:, 0], axis=1).mean()
            calib_h = calib_w * (cy - 1) / (cx - 1)  # Ensure squares have equal x,y side lengths
            obj_4corners = np.array([(-calib_w / 2, -calib_h / 2), (calib_w / 2, -calib_h / 2),
                                     (-calib_w / 2, calib_h / 2), (calib_w / 2, calib_h / 2)], dtype=np.float32) + corners.mean(axis=0)
            # get the perspective transformation matrix (extrinsic)
            self.PerspectiveTransform = cv2.getPerspectiveTransform(img_4corners, obj_4corners)
            corners = cv2.perspectiveTransform(corners[:, None, :], self.PerspectiveTransform)

        if square_side_length:
            corners_grid = corners.reshape((cy, cx, 2))
            side_length_x = np.linalg.norm(np.diff(corners_grid, axis=0), axis=-1).mean()
            side_length_y = np.linalg.norm(np.diff(corners_grid, axis=1), axis=-1).mean()
            self.PixelsPerMm = (side_length_x / 2 + side_length_y / 2) / square_side_length

        image = self.undistort(image)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        image = cv2.drawChessboardCorners(image, (cx, cy), corners, corners_found)
        return image

    def undistort(self, image):
        """ This function undistorts the raw image """
        image = cv2.undistort(image, self.CameraMatrix, self.DistortionCoefficients)
        if self.PerspectiveTransform is not None:
            h, w = image.shape
            image = cv2.warpPerspective(image, self.PerspectiveTransform, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return image

    def import_from_config_file(self, config_path):
        self.config.read(config_path)
        if not self.config.has_section('Calibration'):
            print("Config file has no calibration section to import.")
            return
        self.DistortionCoefficients = np.array(json.loads(self.config['Calibration']['Distortion_Coefficients']))
        self.CameraMatrix = np.array(json.loads(self.config['Calibration']['Camera_Matrix']))
        if self.config.has_option('Calibration', 'Perspective_Transform'):
            self.PerspectiveTransform = np.array(json.loads(self.config['Calibration']['Perspective_Transform']))
        if self.config.has_option('Calibration', 'Pixels_Per_Mm'):
            self.PixelsPerMm = float(self.config['Calibration']['Pixels_Per_Mm'])

    def export_to_config_file(self, config_path):
        self.config.read(config_path)
        self.config['Calibration'] = {"Distortion_Coefficients": self.DistortionCoefficients.ravel().tolist(),
                                      "Camera_Matrix": self.CameraMatrix.tolist()}
        if self.PerspectiveTransform is not None:
            self.config['Calibration']['Perspective_Transform'] = str(self.PerspectiveTransform.tolist())
        if self.PixelsPerMm is not None:
            self.config['Calibration']['Pixels_Per_Mm'] = str(self.PixelsPerMm)
        with open(config_path, "w") as configfile:
            self.config.write(configfile)


if __name__ == '__main__':
    cam = CameraWrapper()
    cam.live_view()
    # calibrator = CheckerboardCalibrator()
    # calibrator.calibrate_from_checkerboard(cv2.imread(r"calibration_board.png", cv2.IMREAD_GRAYSCALE), True, 10)
