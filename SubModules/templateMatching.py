import cv2
import numpy as np
import os


def align_image(image, template):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find the key points and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    # FLANN parameters and matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]

    # Ratio test as per Lowe's paper
    good = []
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.8 * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Use homography to warp image
        height, width, _ = template.shape
        aligned_image = cv2.warpPerspective(image, h, (template.shape[1], template.shape[0]),
                                            borderValue=(255, 255, 255))

        # Identify the black border and fill with white
        mask = aligned_image == 0
        aligned_image[mask] = 255

        return aligned_image
    else:
        print("Not enough matches are found - {}/{}".format(len(good), 10))
        return image  # Return original image if not enough matches


def process_images(source_folder, output_folder, template_path):
    template = cv2.imread(template_path)

    for image_file in os.listdir(source_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_folder, image_file)
            image = cv2.imread(image_path)

            aligned_image = align_image(image, template)

            if aligned_image is not None:
                output_path = os.path.join(output_folder, image_file)
                cv2.imwrite(output_path, aligned_image)
                print(f"Processed and aligned {image_file}")

#
# source_folder = 'data/Scans-4-2-24'
# output_folder = 'data/alignedImages'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# template_path = 'template.jpg'
#
# process_images(source_folder, output_folder, template_path)