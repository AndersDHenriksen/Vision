import time
from pathlib import Path
from pypylon import pylon


class CameraWrapper:

    def __init__(self, exposure_time_us=None, setup_for_streaming=False, enable_jumbo_frame=False):
        self.software_trigger = None
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        if enable_jumbo_frame and self.camera.IsGigE():
            self.camera.GevSCPSPacketSize.SetValue(8192)
            # self.camera.GevSCPD.SetValue(1000)
            # self.camera.GevSCFTD.SetValue(1000)
        if exposure_time_us is not None:
            try:
                self.camera.ExposureTimeAbs = exposure_time_us
            except:
                self.camera.ExposureTime = exposure_time_us
        if setup_for_streaming:
            self.setup_for_streaming()
        else:
            self.setup_for_trigger()
        self.converter = None
        if self.camera.PixelFormat.Value == 'BayerRG8':
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed

    def setup_for_trigger(self):
        self.camera.TriggerSelector = "FrameStart"
        self.camera.TriggerMode = "On"
        self.camera.TriggerSource = "Software"
        self.software_trigger = True

    def setup_for_streaming(self):
        self.camera.TriggerMode = "Off"
        self.software_trigger = False

    def grab(self, wait_for_image=True):
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
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
        import cv2
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


if __name__ == '__main__':
    cam = CameraWrapper()
    cam.live_view()
