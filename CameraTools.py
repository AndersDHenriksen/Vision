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
