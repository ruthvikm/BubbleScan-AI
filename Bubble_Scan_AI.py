import json
import re
import shutil
import fitz  # PyMuPDF
import os
import cv2
import numpy as np


class Scantron95945:
    def __init__(self, pdf_path):
        self.pdf_name = None
        self.pdf_path = pdf_path
        self.source_folder = "data"
        self.output_folder = "data"
        self.template_path = "template.jpg"
        self.extractImagesFromPdf()
        self.template_matching()
        self.extractROIs()
        self.extract_responses()
        shutil.rmtree("data")

    def extractImagesFromPdf(self):

        # Open the PDF file
        pdf_document = fitz.open(self.pdf_path)
        print("------Extracting all the Images from PDF------")

        # Determine PDF name for creating a sub-folder
        self.pdf_name = os.path.splitext(os.path.basename(self.pdf_path))[0]

        # Create a sub-folder for the PDF
        pdf_folder = os.path.join(self.output_folder, self.pdf_name)
        os.makedirs(pdf_folder, exist_ok=True)

        # Iterate over pages and save them as images
        for page_number, page in enumerate(pdf_document):
            # Adjust the page number for naming
            page_number += 1

            image_filename = f"Image_{page_number}.jpg"

            # Get the pixmap of the current page
            original_pix = page.get_pixmap(matrix=fitz.Identity, colorspace=fitz.csRGB, clip=None, annots=True)

            # Calculate scaling factors
            scale_x = 1540 / original_pix.width
            scale_y = 2000 / original_pix.height

            # Apply scaling
            matrix = fitz.Matrix(scale_x, scale_y)

            # Get the scaled pixmap
            pix = page.get_pixmap(matrix=matrix, dpi=300, colorspace=fitz.csRGB, clip=None, annots=True)

            # Save the image
            image_path = os.path.join(pdf_folder, image_filename)
            pix.save(image_path)
            print(f"Extracted {image_filename}")

        pdf_document.close()

    def align_image(self, image, template):
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

    def template_matching(self):
        print("------Template Matching------")
        template = cv2.imread(self.template_path)
        folder = os.path.join(self.source_folder, self.pdf_name)

        # Get a list of image files
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Sort the files; consider natural sorting if filenames have numbers
        sorted_image_files = sorted(image_files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        output_dir = os.path.join(self.output_folder, "alignedImages")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_file in sorted_image_files:
            image_path = os.path.join(folder, image_file)
            image = cv2.imread(image_path)

            aligned_image = self.align_image(image, template)

            if aligned_image is not None:
                output_path = os.path.join(output_dir, image_file)
                cv2.imwrite(output_path, aligned_image)
                print(f"Aligned {image_file}")

    def crop_roi(self, image_path, output_folder):
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
            i = i + 1
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

    def extractROIs(self):
        folder = os.path.join(self.source_folder, "alignedImages")
        # Get a list of image files and sort them by name
        image_files = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sorted_image_files = sorted(image_files, key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])

        output_folder = os.path.join(self.output_folder, "ROIs")
        # Ensure the output directory exists
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for image_file in sorted_image_files:
            image_path = os.path.join(folder, image_file)
            self.crop_roi(image_path, output_folder)

        print("------Extracted all the ROI's------")

    def process_bubble_row(self, image, num_choices=5):
        bubble_width = image.shape[1] // num_choices
        max_white_pixels = 0
        filled_bubble_index = None
        filled_bubble_count = 0  # To count the number of filled bubbles

        # Convert the image to grayscale and apply a threshold to get a binary image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

        min_white_pixels_to_fill = binary.size // num_choices * 0.4

        for i in range(num_choices):
            bubble = binary[:, i * bubble_width:(i + 1) * bubble_width]
            white_pixels = cv2.countNonZero(bubble)

            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                filled_bubble_index = i

            # Check if the bubble is considered filled
            if white_pixels >= min_white_pixels_to_fill:
                filled_bubble_count += 1

        # Return None if more than one bubble is filled or if no bubble is sufficiently filled
        if filled_bubble_count != 1:
            return None

        return chr(ord('A') + filled_bubble_index) if filled_bubble_index is not None else None

    def find_rows(self, image):

        # Convert the image to grayscale and apply Gaussian blur to reduce noise
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Use OTSU method to apply adaptive thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Calculate the vertical projection of the binary image
        vertical_projection = np.sum(binary, axis=1)

        # Detect the row breaks where the sum of the projection falls below a threshold
        row_breaks = np.where(vertical_projection < np.max(vertical_projection) * 0.1)[0]

        # Initialize row boundaries list and add a boundary if the first row starts at the top
        row_boundaries = []
        if row_breaks[0] != 0:
            row_boundaries.append((0, row_breaks[0]))

        # Iterate through row breaks and append boundaries for each detected row
        for start, end in zip(row_breaks, row_breaks[1:]):
            if end - start > 1:  # Check for a significant gap
                row_boundaries.append((start, end))

        # Add a boundary if the last row ends at the bottom
        if row_breaks[-1] != len(vertical_projection) - 1:
            row_boundaries.append((row_breaks[-1], len(vertical_projection) - 1))

        return row_boundaries

    def process_roi(self, image, start_question_num, num_choices=5):
        responses = {}
        row_boundaries = self.find_rows(image)

        for i, (row_start, row_end) in enumerate(row_boundaries):
            question_num = start_question_num + i

            # Ensure the row_end is within the image bounds
            row_end = min(row_end, image.shape[0])

            # Extract the row image based on the identified boundaries
            row = image[row_start:row_end, :]

            # Process the extracted row to find the filled bubble
            response = self.process_bubble_row(row, num_choices)
            responses[f'Q{question_num}'] = response

        return responses

    def process_bubble_column(self, column, num_bubbles=10):
        max_white_pixels = 0
        filled_bubble_index = None

        # Calculate the height of each bubble area within the column
        bubble_height = column.shape[0] // num_bubbles

        for i in range(num_bubbles):
            # Extract the bubble area within the column
            bubble = column[i * bubble_height:(i + 1) * bubble_height, :]

            # Count the number of white pixels in the bubble area
            white_pixels = cv2.countNonZero(bubble)

            if white_pixels > max_white_pixels:
                max_white_pixels = white_pixels
                filled_bubble_index = i

        min_white_pixels_to_fill = column.size // num_bubbles * 0.3

        if max_white_pixels < min_white_pixels_to_fill:
            return None
        # The filled_bubble_index corresponds to the digit
        return filled_bubble_index

    def process_student_id(self, roi, num_columns=10, num_bubbles=10):
        student_id = ''

        # Calculate the width of each digit's column in the ROI
        digit_width = roi.shape[1] // num_columns

        # Process each column to identify the filled bubble
        for i in range(num_columns):
            # Extract the column ROI
            column_roi = roi[:, i * digit_width:(i + 1) * digit_width]

            # Convert the column ROI to grayscale and threshold it to create a binary image
            gray = cv2.cvtColor(column_roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

            # Process the column to determine the filled bubble
            filled_digit = self.process_bubble_column(binary, num_bubbles)

            # Append the filled digit to the student ID, or 'X' if not detected
            student_id += str(filled_digit) if filled_digit is not None else 'X'

        return student_id

    def extract_responses(self):
        students_results = []
        base_folder_path = os.path.join(self.source_folder, "ROIs")

        # Get all student image folders and sort them to maintain order
        # student_image_folders = sorted(os.listdir(base_folder_path))
        student_image_folders = sorted(os.listdir(base_folder_path), key=lambda x: [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', x)])
        for student_image_folder in student_image_folders:
            student_folder_path = os.path.join(base_folder_path, student_image_folder)
            if os.path.isdir(student_folder_path):
                # Assuming file names are consistent with 'first', 'second', 'studentid' identifiers
                first_column_path = os.path.join(student_folder_path, 'first_column_roi.jpg')
                second_column_path = os.path.join(student_folder_path, 'second_column_roi.jpg')
                student_id_path = os.path.join(student_folder_path, 'student_id_roi.jpg')

                if not all([os.path.exists(first_column_path), os.path.exists(second_column_path), os.path.exists(student_id_path)]):
                    print(f"Missing ROI images in folder: {student_folder_path}")
                    continue

                # Process images in the order they are found
                student_id = self.process_student_id(cv2.imread(student_id_path))
                responses_first = self.process_roi(cv2.imread(first_column_path), start_question_num=1)
                responses_second = self.process_roi(cv2.imread(second_column_path), start_question_num=26)

                student_data = {
                    "studentID": student_id,
                    "answers": {**responses_first, **responses_second}
                }
                students_results.append(student_data)

        final_output = {"students": students_results}

        with open('result_data.json', 'w') as json_file:
            json.dump(final_output, json_file, indent=4)

        print("Processing complete. Data saved to 'result_data.json'.")

