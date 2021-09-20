from pathlib import Path
import time
import numpy as np
import cv2

times_spent = []


class Timing:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        time_spend = (time.time() - self.start_time) * 1000
        times_spent.append(time_spend)
        print(f"{self.name}: {time_spend:.0f} µs")


def time_normal_vision_operations():
    with Timing(" Image generation"):
        u, v = np.meshgrid(np.arange(4000), np.arange(4000))
        images = np.array([np.random.randint(50, 200, (4000, 4000)), np.eye(4000), u % 255, v // 20]).astype(np.uint8)

    with Timing("Template matching"):
        template = u[:100, :100].astype(np.uint8)
        for img in images:
            xcor_out = cv2.matchTemplate(img, template, cv2.TM_CCORR)

    with Timing("  Threshold + any"):
        for _ in range(4):
            for img in images:
                bw_img = np.bitwise_xor(img > 100, img > 200)
                out = bw_img.any(axis=0).sum() + bw_img.any(axis=1).sum()

    with Timing("       Morphology"):
        for _ in range(3):
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            kernel25 = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            for img in images:
                dilate_out = cv2.dilate(img, kernel)
                hat_out = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
                dilate_out25 = cv2.dilate(img, kernel25)
                hat_out25 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel25)

    with Timing("             Diff"):
        for _ in range(2):
            for img in images:
                diff_img = np.diff(images, axis=0)
                out_img = images[:-1, :] - diff_img

    with Timing("       Conversion"):
        for _ in range(4):
            for img in images:
                img_float = img.astype(np.float32)
                img_int = img.astype(int)

    with Timing("          Disc IO"):
        for img in images:
            temp_path = Path("temp.png")
            cv2.imwrite(str(temp_path), img)
            img2 = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)
            out = np.all(img == img2)
            temp_path.unlink()


if __name__ == "__main__":
    for _ in range(10):
        time_normal_vision_operations()
    print("---------------------------")
    print(f"   Overall scores: {60000 / np.mean(times_spent):.0f} pts")


# AHE laptop: 95 pts
# Vision desktop: 109 pts
# Raspberry Pi4: 14 pts
