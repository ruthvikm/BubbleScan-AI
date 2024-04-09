import cv2
import numpy as np
import os


# def align_image(image, template):
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()
#
#     # Find the key points and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(image, None)
#     kp2, des2 = sift.detectAndCompute(template, None)
#
#     # FLANN parameters and matcher
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     # Need to draw only good matches, so create a mask
#     matchesMask = [[0, 0] for i in range(len(matches))]
#
#     # Ratio test as per Lowe's paper
#     good = []
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.8 * n.distance:
#             matchesMask[i] = [1, 0]
#             good.append(m)
#
#     if len(good) > 10:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#         # Find homography
#         h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#         # Use homography to warp image
#         height, width, _ = template.shape
#         aligned_image = cv2.warpPerspective(image, h, (template.shape[1], template.shape[0]),
#                                             borderValue=(255, 255, 255))
#
#         # Identify the black border and fill with white
#         mask = aligned_image == 0
#         aligned_image[mask] = 255
#
#         return aligned_image
#     else:
#         print("Not enough matches are found - {}/{}".format(len(good), 10))
#         return image  # Return original image if not enough matches

# def align_image(self, image, template):
#     # Initialize SIFT detector
#     sift = cv2.SIFT_create()
#
#     # Find the key points and descriptors with SIFT
#     kp1, des1 = sift.detectAndCompute(image, None)
#     kp2, des2 = sift.detectAndCompute(template, None)
#
#     # FLANN parameters and matcher
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     # Need to draw only good matches, so create a mask
#     matchesMask = [[0, 0] for i in range(len(matches))]
#
#     # Ratio test as per Lowe's paper
#     good = []
#     for i, (m, n) in enumerate(matches):
#         if m.distance < 0.8 * n.distance:
#             matchesMask[i] = [1, 0]
#             good.append(m)
#
#     if len(good) > 10:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#         # Find homography
#         h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#         # Use homography to warp image
#         height, width, _ = template.shape
#         aligned_image = cv2.warpPerspective(image, h, (template.shape[1], template.shape[0]),
#                                             borderValue=(255, 255, 255))
#
#         # Identify the black border and fill with white
#         mask = aligned_image == 0
#         aligned_image[mask] = 255
#
#         return aligned_image
#     else:
#         print("Not enough matches are found - {}/{}".format(len(good), 10))
#         return image  # Return original image if not enough matches

# def align_image(self, image, template):
#     # Initialize ORB detector
#     orb = cv2.ORB_create(nfeatures=1000)  # You can adjust `nfeatures` based on your needs
#
#     # Find the key points and descriptors with ORB
#     kp1, des1 = orb.detectAndCompute(image, None)
#     kp2, des2 = orb.detectAndCompute(template, None)
#
#     # Check if descriptors were found
#     if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
#         print("Not enough keypoints are found in one or both images.")
#         return image
#
#     # FLANN parameters and matcher, adjusted for ORB
#     FLANN_INDEX_LSH = 6
#     index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
#     search_params = dict(checks=100)  # Increase checks for better accuracy
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(des1, des2, k=2)
#
#     # Ratio test as per Lowe's paper
#     good = []
#     for m, n in matches:
#         if m.distance < 0.75 * n.distance:  # Adjust the ratio as needed
#             good.append(m)
#
#     if len(good) > 10:
#         src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#         dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#
#         # Find homography
#         h, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#
#         if h is None:
#             print("Homography could not be computed.")
#             return image
#
#         # Use homography to warp image
#         height, width, _ = template.shape
#         aligned_image = cv2.warpPerspective(image, h, (width, height), borderValue=(255, 255, 255))
#
#         # Identify the black border and fill with white (this step can be adjusted based on your needs)
#         mask = aligned_image == 0
#         aligned_image[mask] = 255
#
#         return aligned_image
#     else:
#         print(f"Not enough good matches are found - {len(good)}/{10}")
#         return image  # Return original image if not enough matches

def align_image(self, image, template):
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=10000)

    # Find the key points and descriptors with ORB
    kp1, des1 = orb.detectAndCompute(image, None)
    kp2, des2 = orb.detectAndCompute(template, None)

    # Ensure there are enough descriptors to match
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        print("Not enough descriptors.")
        return image

    # FLANN parameters and matcher, adjusted for ORB
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio test as per Lowe's paper
    good = []
    for match in matches:
        if len(match) == 2:  # Ensure there are 2 matches to unpack
            m, n = match
            if m.distance < 0.8 * n.distance:
                good.append(m)

    if len(good) > 10:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Find homography
        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if h is None:
            print("Homography computation failed.")
            return image

        # Use homography to warp image
        height, width, _ = template.shape
        aligned_image = cv2.warpPerspective(image, h, (width, height), borderValue=(255, 255, 255))

        # Optional: remove the black border by filling it with white
        mask = aligned_image == 0
        aligned_image[mask] = 255

        return aligned_image
    else:
        print("Not enough good matches are found - {}/{}".format(len(good), 10))
        return image


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