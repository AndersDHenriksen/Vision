from pypylon import pylon


class CameraWrapper:

    def __init__(self, exposure_time_us=3000, grab_strategy='latest'):
        assert grab_strategy in ['latest', 'upcoming']
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        self.camera.Open()
        self._grab_strategy = grab_strategy
        if self.camera.IsGigE():
            self.camera.ExposureTimeAbs = exposure_time_us
        else:
            self.camera.ExposureTime = exposure_time_us

    def grab(self):
        if not self.camera.IsGrabbing():
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly if self._grab_strategy == 'latest' else
                                      pylon.GrabStrategy_UpcomingImage)
        grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        while not grabResult.GrabSucceeded():
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
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
