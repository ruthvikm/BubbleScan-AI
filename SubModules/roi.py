import os

import cv2


def crop_columns_based_on_markers(image_path, output_folder):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the regions of interest as the top and left parts of the image
    height, width = gray.shape
    top_region_height = int(height * 0.03)
    left_region_width = int(width * 0.04)
    top_region_width = int(width * 0.03)
    left_region_height = int(height * 0.04)

    # Threshold the entire image first to get potential markers
    _, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the threshold image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    top_markers = []
    left_markers = []
    i = 0
    # Collect markers within the specified region
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        i = i+1
        # Top region markers
        if y < top_region_height and x >= top_region_width and 200 <= w * h <= 5000:
            top_markers.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Left region markers
        elif x < left_region_width and y >= left_region_height and 100 <= w * h <= 5000:
            left_markers.append((x, y, w, h))
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Sort the markers
    top_markers.sort(key=lambda m: m[0])  # Sort by x (left to right)
    left_markers.sort(key=lambda m: m[1])  # Sort by y (top to bottom)

    # Get the required markers for cropping
    top_marker_1 = top_markers[0] if len(top_markers) > 1 else None
    top_marker_3 = top_markers[2] if len(top_markers) > 3 else None
    left_marker_5 = left_markers[5] if len(left_markers) > 4 else None
    left_marker_53 = left_markers[53] if len(left_markers) > 52 else None
    top_marker_4 = top_markers[3] if len(top_markers) > 3 else None
    left_marker_34 = left_markers[34] if len(left_markers) > 34 else None
    left_marker_43 = left_markers[43] if len(left_markers) > 42 else None

    # Check if all necessary markers are found
    if not all([top_marker_1, top_marker_3, left_marker_5, left_marker_53]):
        print("Not all markers found, cannot crop image accurately.")
        return

    # Calculate the ROI using the intersection of these markers for the first column
    x1_first_column = top_marker_1[0]
    x2_first_column = top_marker_1[0] + 160
    y1_columns = left_marker_5[1] - 10
    y2_columns = left_marker_53[1] + left_marker_53[3] + 10

    # Crop the image for the first column
    first_column_roi = image[y1_columns:y2_columns, x1_first_column:x2_first_column]

    # Calculate the ROI using the intersection of these markers for the second column
    x1_second_column = x2_first_column + 100  # start of the second column is the end of the first column
    x2_second_column = top_marker_3[0] + top_marker_3[2] + 10
    y1_columns = left_marker_5[1] + 10
    y2_columns = left_marker_53[1] + left_marker_53[3] + 35

    # Crop the image for the second column
    second_column_roi = image[y1_columns:y2_columns, x1_second_column:x2_second_column]

    # Calculate the ROI for the student ID section
    x1_student_id = top_marker_4[0] - 130
    x2_student_id = top_marker_4[0] + top_marker_4[2] + 165
    y1_student_id = left_marker_34[1] + 20
    y2_student_id = left_marker_43[1] + left_marker_43[3] + 30

    student_id_roi = image[y1_student_id:y2_student_id, x1_student_id:x2_student_id]

    # Create a unique output folder for this image
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = os.path.join(output_folder, image_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Construct the full path for saving each ROI
    first_column_path = os.path.join(output_folder, "first_column_roi.jpg")
    second_column_path = os.path.join(output_folder, "second_column_roi.jpg")
    student_id_path = os.path.join(output_folder, "student_id_roi.jpg")

    # Save the cropped images
    cv2.imwrite(student_id_path, student_id_roi)
    cv2.imwrite(first_column_path, first_column_roi)
    cv2.imwrite(second_column_path, second_column_roi)

    return first_column_path, second_column_path, student_id_path


def process_all_images(source_folder, output_base_path):
    for image_file in os.listdir(source_folder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(source_folder, image_file)
            first_column_path, second_column_path, student_id_path = crop_columns_based_on_markers(image_path, output_base_path)
            print(f"Processed {image_file}:")
            print(f"  First column ROI: {first_column_path}")
            print(f"  Second column ROI: {second_column_path}")
            print(f"  Student ID ROI: {student_id_path}")


# Paths to the source folder and the base output folder
# source_folder = 'data/alignedImages'
# output_base_path = 'data/ROIs'
# if not os.path.exists(output_base_path):
#     os.makedirs(output_base_path)
# process_all_images(source_folder, output_base_path)
