from pathlib import Path
import numpy as np
import Vision.VisionTools as vt
from Vision.DecoratorTools import path2image


@path2image(load_from_shared_memory=False, enforce_grayscale=True)
def inspect(image):
    return False
    # vt.showimg(image)


def data_result_generator():
    rejects = sorted(Path(r"").rglob("*.png"))  # TODO fill in this
    accepts = sorted(Path(r"").rglob("*.png"))
    for f, image_path in enumerate(rejects):
        yield f, 1, image_path
    for g, image_path in enumerate(accepts):
        yield g + len(rejects), 0, image_path


def test_inspection(idx=None):
    annotations = []
    results = []
    for i, annotation, image_path in data_result_generator():
        if idx is not None and i not in (idx if isinstance(idx, list) else [idx]):
            continue
        annotations.append(annotation)
        results.append(inspect(image_path))
        print(f"{i:3} | {int(annotations[-1] == results[-1])} | {image_path}")
    annotations, results = np.array(annotations), np.array(results)
    errors = annotations != results
    print(f"Reject Errors (FN): {np.sum(errors & (annotations==1))} / {np.sum(annotations==1)} | errors: {vt.find(errors & (annotations==1))}")
    print(f"Accept Errors (FP): {np.sum(errors & (annotations==0))} / {np.sum(annotations==0)} | errors: {vt.find(errors & (annotations==0))}")


if __name__ == "__main__":
    test_inspection()
