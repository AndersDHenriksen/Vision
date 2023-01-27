import time
import concurrent
import numpy as np
import cv2
import VisionTools as vt


class PyramidMatch:
    def __init__(self, model_img, mask_img=None, pyramid_thres=0.9, rotate_thres=0.8, min_angle_resolution_deg=5):
        if mask_img is None:
            mask_img = 255 * np.ones(model_img.shape, 'uint8')

        # Estimate how much we can downsample model
        self.model_pyramid = [model_img]
        self.mask_pyramid = [mask_img]
        while True:
            down_sampled = cv2.pyrDown(self.model_pyramid[-1])
            up_sampled = down_sampled
            for i in range(len(self.model_pyramid)):
                up_sampled = cv2.pyrUp(up_sampled)

            if cv2.matchTemplate(self.model_pyramid[0], up_sampled,
                                 cv2.TM_CCOEFF_NORMED).max() > pyramid_thres:  # Also consider non-perfect alignment?
                self.model_pyramid.append(down_sampled)
                self.mask_pyramid.append(cv2.pyrDown(self.mask_pyramid[-1]))
            else:
                break
        # print(f"Downsampling used: {len(self.model_pyramid) - 1}")

        # Calculate which angles to use
        rot_models = [self.model_pyramid[-1]]
        rot_angles = [0]
        for angle in np.arange(2, 360, 2):
            rot_model = vt.simple_rotate(self.model_pyramid[-1], angle)
            # rot_mask = vt.simple_rotate(mask_pyramid[-1], angle)
            image = rot_models[-1]
            if (dh := rot_model.shape[0] - image.shape[0]) > 0:
                image = cv2.copyMakeBorder(image, dh, dh, 0, 0, cv2.BORDER_REPLICATE)
            if (dw := rot_model.shape[1] - image.shape[1]) > 0:
                image = cv2.copyMakeBorder(image, 0, 0, dw, dw, cv2.BORDER_REPLICATE)
            if match_template(image, rot_model).max() < rotate_thres:
                rot_models.append(rot_model)
                # rot_masks.append(rot_mask)  # Not needed
                rot_angles.append(angle)
        self.d_angle = min(45, min_angle_resolution_deg * 2 ** (len(self.model_pyramid) - 1))
        if len(rot_angles) > 1:
            self.d_angle = min(self.d_angle, np.diff(rot_angles).min())
        self.rot_angles = np.arange(0, 360, self.d_angle)
        self.rot_models = [vt.simple_rotate(self.model_pyramid[-1], angle) for angle in self.rot_angles]
        self.rot_masks = [vt.simple_rotate(self.mask_pyramid[-1], angle) for angle in self.rot_angles]
        # print(f"Angles used: {len(self.rot_angles)}")

    def match(self, scene_img, candidate_thres=0.8, n_out=None):
        rot_models = self.rot_models
        rot_masks = self.rot_masks
        start_time = time.time()

        # Expand scene image so model can be outside scene
        n_expand = int(np.linalg.norm(self.model_pyramid[0].shape) / 3)
        scene_img = cv2.copyMakeBorder(scene_img, n_expand, n_expand, n_expand, n_expand, cv2.BORDER_REPLICATE)

        # downsample scene
        scene_pyramid = [scene_img]
        for i in range(len(self.model_pyramid) - 1):
            scene_pyramid.append(cv2.pyrDown(scene_pyramid[-1]))

        matches = []
        for model, mask, angle in zip(rot_models, rot_masks, self.rot_angles):
            match = match_template(scene_pyramid[-1], model, mask=mask)
            nms_inplace(match, model)
            matches.append(match)
        matches = np.dstack(matches)

        # More nms
        match_idx = matches.argmax(axis=-1)
        match = matches.max(axis=-1)
        nms_inplace(match, self.model_pyramid[-1], True)

        # Check good positions going up pyramid, only around best angle will be used going forward
        candidates_masks = match > candidate_thres
        candidates = []
        for candidate_ij in np.argwhere(candidates_masks):
            i, j = candidate_ij
            angle = self.rot_angles[match_idx[i, j]]
            candidates.append((i, j, angle, match[i, j]))
        candidates = nms_list(candidates, self.model_pyramid[-1].shape)
        da = self.d_angle
        # vt.showimg(plot_boxes(scene_pyramid[-1], model_pyramid[-1], candidates_old, 1))  # debug

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for p in range(2, len(self.model_pyramid) + 1):
                candidates_old = candidates
                do_sub_pixel = do_sub_angel = p == len(self.model_pyramid)
                current_model = self.model_pyramid[-p]
                current_mask = self.mask_pyramid[-p]
                current_scene = scene_pyramid[-p]
                da /= 2

                # Test candidates
                input_list = []
                for candidate_ijas in candidates_old:
                    i, j, angle, _ = candidate_ijas
                    for angle in [angle - da, angle, angle + da]:
                        model = vt.simple_rotate(current_model, angle)
                        mask = vt.simple_rotate(current_mask, angle)
                        i1 = min(current_scene.shape[0] - model.shape[0] - 2, max(0, 2 * i - 1 - model.shape[0] // 2))
                        i2 = i1 + model.shape[0] + 2
                        j1 = min(current_scene.shape[1] - model.shape[1] - 2, max(0, 2 * j - 1 - model.shape[1] // 2))
                        j2 = j1 + model.shape[1] + 2
                        scene_crop = current_scene[i1:i2, j1:j2]
                        input_list.append((scene_crop, model, mask, angle, i1, j1, do_sub_pixel))
                        # tm((scene_crop, model, mask, angle, i1, j1)) # For debug only

                # candidates_all = (tm(inp) for inp in input_list)  # use this line to not use multithread
                candidates_all = executor.map(tm, input_list)
                candidates = []
                for _ in range(len(candidates_old)):
                    angle_candidates = [next(candidates_all), next(candidates_all), next(candidates_all)]
                    candidates.append(max(angle_candidates, key=lambda c: c[3]))
                    if do_sub_angel:
                        angles = np.array([c[2] for c in angle_candidates])
                        scores = np.array([c[3] for c in angle_candidates])
                        if scores.argmax() == 1:
                            new_angle = np.interp(vt.weighted_3point_extrema(scores), np.arange(3), angles)
                            candidates[-1] = (candidates[-1][0], candidates[-1][1], new_angle, candidates[-1][3])

                candidates = [c for c in candidates if c[3] > candidate_thres]
                candidates = nms_list(candidates, current_model.shape)
                # print(f"Level {p} done after {time.time() - start_time:.2f} sec. {len(candidates)} candidates left")
                if not len(candidates):
                    break
                # vt.showimg(plot_boxes(current_scene, current_model, candidates))

        if n_out is not None:
            candidates = candidates[:n_out]
        candidates = [(c[0] - n_expand, c[1] - n_expand, c[2], c[3]) for c in candidates]
        # print(f"Refine search done after {time.time() - start_time:.2f} sec")
        return candidates
        vt.showimg(plot_boxes(scene_img0, self.model_pyramid[0], candidates))


def match_template(image, template, mask=None, method=cv2.TM_CCORR_NORMED):
    """ Wrapper for cv2.matchTemplate but output is same size as input image """
    mt_out = cv2.matchTemplate(image, template, method, mask=mask)
    th, tw = template.shape
    return cv2.copyMakeBorder(mt_out, th//2, th - th//2 - 1, tw//2, tw - tw//2 - 1, cv2.BORDER_CONSTANT)


def plot_boxes(scene, model, candidates, thickness=3):
    draw = cv2.cvtColor(scene, cv2.COLOR_GRAY2BGR)
    h2, w2 = model.shape[0]/2, model.shape[1]/2
    all_pts = []
    for candidate_ija in candidates:
        i, j, a, _ = candidate_ija
        m = cv2.getRotationMatrix2D((0, 0), -a, 1)
        m[:, 2] = [j, i]
        pts = [vt.intr(np.dot(m, [a, b, 1])) for a, b in zip([w2, w2, -w2, -w2], [h2, -h2, -h2, h2])]
        all_pts.append(pts)
    draw = cv2.polylines(draw, np.int32(all_pts), True, (255, 0, 0), thickness=thickness)
    return draw


def nms_inplace(match, model, use_smallest_dim=False):
    size = np.array(model.shape)[::-1] // 2 + 1
    if use_smallest_dim:
        size[:] = size.min()
    match[match < vt.morph('dilate', match, size)] = 0  # NMS


def nms_list(candidates, model_shape):
    min_d = min(model_shape)
    max_d = max(model_shape)
    n_candidates = len(candidates)
    if n_candidates < 2:
        return candidates
    cand = np.array(candidates)
    cand[:, 2] = np.deg2rad(cand[:, 2])
    n_discard = 0
    candidates_new = []
    while n_candidates > len(candidates_new) + n_discard:
        max_idx = cand[:, 3].argmax()
        candidates_new.append(candidates[max_idx])
        keep_ij = cand[max_idx, :2].copy()
        keep_a = cand[max_idx, 2]
        cand[max_idx, :] = -np.inf
        for i in np.flatnonzero(np.linalg.norm(cand[:, :2] - keep_ij, axis=1) < max_d):
            a = np.arctan((keep_ij[0] - cand[i, 0]) / (keep_ij[1] - cand[i, 1] + 1e-6))
            d = (np.abs(np.cos(keep_a - a)) + np.abs(np.cos(cand[i, 2] - a))) * (max_d - min_d) / 2 + min_d
            if np.linalg.norm(cand[i, :2] - keep_ij) < 0.8 * d:
                n_discard += 1
                cand[i, :] = -np.inf
    return candidates_new


def tm(all_inputs, use_fast_mask=False):
    scene_crop, model, mask, angle, i1, j1, do_sub_pixel = all_inputs
    if use_fast_mask:
        scene_crop2 = scene_crop.copy()
        scene_crop2[cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT) < 128] = 0
        mt_out = cv2.matchTemplate(scene_crop2, model, cv2.TM_CCOEFF_NORMED)
    else:
        mt_out = cv2.matchTemplate(scene_crop, model, cv2.TM_CCOEFF_NORMED, mask=mask)  # TM_CCORR_NORMED faster
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(mt_out)
    th, tw = model.shape
    if not do_sub_pixel:
        return maxLoc[1] + th // 2 + i1, maxLoc[0] + tw // 2 + j1, angle, maxVal
    dj, di = maxLoc
    if di == 1:
        di = vt.weighted_3point_extrema(mt_out[:, dj])
    if dj == 1:
        dj = vt.weighted_3point_extrema(mt_out[maxLoc[1], :])
    return di + th // 2 + i1, dj + tw // 2 + j1, angle, maxVal


def test_pyramid_match(scene_img, model_img):
    matcher = PyramidMatch(model_img)
    candidates = matcher.match(scene_img)
    print(candidates)
    vt.showimg(plot_boxes(scene_img, model_img, candidates), close_on_click=True)


if __name__ == "__main__":
    pass
