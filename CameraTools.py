from pypylon import pylon


class CameraWrapper:

    def __init__(self, exposure_time_us=3000, setup_for_streaming=False, enable_jumbo_frame=True):
        self.software_trigger = None
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        if enable_jumbo_frame and self.camera.IsGigE():
            self.camera.GevSCPSPacketSize.SetValue(8192)
        try:
            self.camera.ExposureTimeAbs = exposure_time_us
        except:
            self.camera.ExposureTime = exposure_time_us
        if setup_for_streaming:
            self.setup_for_streaming()
        else:
            self.setup_for_trigger()

    def setup_for_trigger(self):
        self.camera.TriggerSelector = "FrameStart"
        self.camera.TriggerMode = "On"
        self.camera.TriggerSource = "Software"
        self.software_trigger = True

    def setup_for_streaming(self):
        self.camera.TriggerMode = "Off"
        self.software_trigger = False

    def grab(self):
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing()
        if self.software_trigger:
            self.camera.ExecuteSoftwareTrigger()
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        while not grabResult.GrabSucceeded():
            if self.software_trigger:
                self.camera.ExecuteSoftwareTrigger()
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            grabResult.Release()
        image = grabResult.GetArray()
        grabResult.Release()
        return image

    def grab_single(self):
        return self.camera.GrabOne(1000).Array

    def stop(self):
        self.camera.StopGrabbing()
        self.camera.Close()

    def live_view(self):
        import cv2
        print("Starting liveview. Controls are:\nspace: pause\nq\esc: quit")
        pause = False
        while True:
            if not pause:
                image = self.grab()
            if image.ndim == 3:
                image = image[:, :, ::-1]
            image = cv2.resize(image, (1200, int(image.shape[0] / image.shape[1] * 1200)))
            cv2.imshow("Live View", image)
            key = cv2.waitKey(10)
            if key in [27, 113]:
                break
            if key == 32:
                pause = not pause


if __name__ == '__main__':
    cam = CameraWrapper()
    cam.live_view()
