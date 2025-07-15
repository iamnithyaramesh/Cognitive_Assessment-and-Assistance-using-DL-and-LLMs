import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os

def preprocess(img_path, resize_to=(300, 100)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if resize_to:
        img = cv2.resize(img, resize_to)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    return img, thresh

def get_largest_contour(thresh_img):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv2.contourArea) if contours else None

def loop_count_heuristic(thresh_img, original):
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    count = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 100 and len(cnt) >= 5:
            ellipse = cv2.fitEllipse(cnt)
            _, axes, _ = ellipse
            major, minor = max(axes), min(axes)
            ecc = np.sqrt(1 - (minor / major) ** 2)
            if 0.3 < ecc < 0.9:
                count += 1
    return count

def hu_moment_score(contour1, contour2):
    hu1 = cv2.HuMoments(cv2.moments(contour1)).flatten()
    hu2 = cv2.HuMoments(cv2.moments(contour2)).flatten()
    score = np.sum(np.abs(-np.sign(hu1) * np.log10(np.abs(hu1)) -
                          -np.sign(hu2) * np.log10(np.abs(hu2))))
    return score

def orb_similarity(img1, img2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    return len(matches) / max(len(kp1), len(kp2)) if kp1 and kp2 else 0

def combined_score( test_path,ref_path=r"test_data\double_infinity test images\ref.png", show=False):
    ref_img, ref_thresh = preprocess(ref_path)
    test_img, test_thresh = preprocess(test_path)

    # SSIM
    ssim_score, _ = ssim(ref_thresh, test_thresh, full=True)

    # Contours
    ref_cont = get_largest_contour(ref_thresh)
    test_cont = get_largest_contour(test_thresh)
    shape_score = cv2.matchShapes(ref_cont, test_cont, cv2.CONTOURS_MATCH_I1, 0.0)

    # Loop counts
    loop_ref = loop_count_heuristic(ref_thresh, ref_img)
    loop_test = loop_count_heuristic(test_thresh, test_img)
    loop_diff = abs(loop_ref - loop_test)

    # Hu Moments
    hu_score = hu_moment_score(ref_cont, test_cont)

    # ORB Similarity
    orb_score = orb_similarity(ref_img, test_img)

    # Normalize + weight scoring (lower = better except for orb & ssim)
    score = (
        (1 - ssim_score) * 0.25 +
        (shape_score) * 0.25 +
        (loop_diff / 4.0) * 0.15 +  # assuming max difference of 4
        (hu_score / 10.0) * 0.2 +   # scaled to normalize
        (1 - orb_score) * 0.15
    )

    result = 1 if score < 0.45 else 0  # threshold can be adjusted

    if show:
        print(f"[INFO] SSIM: {ssim_score:.3f}, Shape: {shape_score:.3f}, "
              f"LoopΔ: {loop_diff}, Hu: {hu_score:.3f}, ORB: {orb_score:.3f} -> Final Score: {score:.3f} => Result: {result}")
        combined = np.hstack((cv2.cvtColor(ref_thresh, cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(test_thresh, cv2.COLOR_GRAY2BGR)))
        plt.imshow(combined)
        plt.title(f"Result: {result}")
        plt.axis("off")
        plt.show()

    return result



