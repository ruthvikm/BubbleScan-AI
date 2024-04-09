import os

import cv2
import json
import numpy as np


# def process_bubble_row(image, num_choices=5):
#     bubble_width = image.shape[1] // num_choices
#     max_white_pixels = 0
#     filled_bubble_index = None
#     filled_bubble_count = 0  # To count the number of filled bubbles
#
#     # Convert the image to grayscale and apply a threshold to get a binary image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#
#     min_white_pixels_to_fill = binary.size // num_choices * 0.3
#
#     for i in range(num_choices):
#         bubble = binary[:, i * bubble_width:(i + 1) * bubble_width]
#         white_pixels = cv2.countNonZero(bubble)
#
#         if white_pixels > max_white_pixels:
#             max_white_pixels = white_pixels
#             filled_bubble_index = i
#
#         # Check if the bubble is considered filled
#         if white_pixels >= min_white_pixels_to_fill:
#             filled_bubble_count += 1
#
#     # Return None if more than one bubble is filled or if no bubble is sufficiently filled
#     if filled_bubble_count != 1:
#         return None
#
#     return chr(ord('A') + filled_bubble_index) if filled_bubble_index is not None else None

# def process_bubble_row(self, image, num_choices=5):
#     bubble_width = image.shape[1] // num_choices
#     max_white_pixels = 0
#     filled_bubble_index = None
#     filled_bubble_count = 0  # To count the number of filled bubbles
#
#     # Convert the image to grayscale and apply a threshold to get a binary image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#
#     min_white_pixels_to_fill = binary.size // num_choices * 0.3
#
#     for i in range(num_choices):
#         bubble = binary[:, i * bubble_width:(i + 1) * bubble_width]
#         white_pixels = cv2.countNonZero(bubble)
#
#         if white_pixels > max_white_pixels:
#             max_white_pixels = white_pixels
#             filled_bubble_index = i
#
#         # Check if the bubble is considered filled
#         if white_pixels >= min_white_pixels_to_fill:
#             filled_bubble_count += 1
#
#     # Return None if more than one bubble is filled or if no bubble is sufficiently filled
#     if filled_bubble_count != 1:
#         return None
#
#     return chr(ord('A') + filled_bubble_index) if filled_bubble_index is not None else None


# def process_bubble_row(self, image, num_choices=5):
#     bubble_width = image.shape[1] // num_choices
#     filled_bubbles = []  # To store indices of filled bubbles
#
#     # Convert the image to grayscale and apply a threshold to get a binary image
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
#
#     min_white_pixels_to_fill = binary.size // num_choices * 0.3
#
#     for i in range(num_choices):
#         bubble = binary[:, i * bubble_width:(i + 1) * bubble_width]
#         white_pixels = cv2.countNonZero(bubble)
#
#         # Check if the bubble is considered filled
#         if white_pixels >= min_white_pixels_to_fill:
#             filled_bubbles.append(i)
#
#     # Return the list of filled bubble indices converted to their corresponding letter options
#     return [chr(ord('A') + index) for index in filled_bubbles]
#

def process_bubble_row(self, image, num_choices=5):
    bubble_width = image.shape[1] // num_choices
    filled_bubbles = []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 137, 255, cv2.THRESH_BINARY_INV)

    min_white_pixels_to_fill = binary.size // num_choices * 0.39

    for i in range(num_choices):
        bubble = binary[:, i * bubble_width:(i + 1) * bubble_width]
        white_pixels = cv2.countNonZero(bubble)

        if white_pixels >= min_white_pixels_to_fill:
            filled_bubbles.append(i)

    if len(filled_bubbles) == 0:
        return None
    elif len(filled_bubbles) == 1:
        return chr(ord('A') + filled_bubbles[0])
    else:
        #return [chr(ord('A') + index) for index in filled_bubbles]
        return "multi"



def find_rows_by_projection(image):
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

    # # Create an annotated image with the row boundaries drawn
    # annotated_image = image.copy()
    # for top, bottom in row_boundaries:
    #     cv2.line(annotated_image, (0, top), (image.shape[1], top), (0, 0, 255), 2)
    #     cv2.line(annotated_image, (0, bottom), (image.shape[1], bottom), (0, 255, 0), 2)
    #
    # # Save the annotated image
    # cv2.imwrite('annotated_rows.jpg', annotated_image)

    return row_boundaries


def process_roi(image, start_question_num, num_choices=5):
    responses = {}
    row_boundaries = find_rows_by_projection(image)

    print(f"Identified row boundaries: {row_boundaries}")

    for i, (row_start, row_end) in enumerate(row_boundaries):
        question_num = start_question_num + i

        # Ensure the row_end is within the image bounds
        row_end = min(row_end, image.shape[0])

        # Extract the row image based on the identified boundaries
        row = image[row_start:row_end, :]

        print(
            f"Processing row {i + 1}, Question {question_num}, Row start: {row_start},"
            f" Row end: {row_end}, Row shape: {row.shape}")

        # Process the extracted row to find the filled bubble
        response = process_bubble_row(row, num_choices)
        responses[f'Q{question_num}'] = response

    return responses


def process_bubble_column(column, num_bubbles=10):
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


def process_student_id(roi, num_columns=10, num_bubbles=10):
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
        filled_digit = process_bubble_column(binary, num_bubbles)

        # Append the filled digit to the student ID, or 'X' if not detected
        student_id += str(filled_digit) if filled_digit is not None else 'X'

    return student_id


# def main():
#     first_roi_image = cv2.imread('croppedImages/Image_1/first_column_roi.jpg')
#     second_roi_image = cv2.imread('croppedImages/Image_1/second_column_roi.jpg')
#     student_id_image = cv2.imread('croppedImages/Image_1/student_id_roi.jpg')
#
#     responses_first = process_roi(first_roi_image, start_question_num=1)
#     responses_second = process_roi(second_roi_image, start_question_num=26)
#     student_id = process_student_id(student_id_image)
#
#     responses = {**responses_first, **responses_second}
#
#     result_data = {
#         'student_id': student_id,
#         'responses': responses
#     }
#
#     with open('result_data.json', 'w') as json_file:
#         json.dump(result_data, json_file, indent=4)
#
#     print("Processing complete. Data saved to 'result_data.json'.")
#
#
# if __name__ == '__main__':
#     main()

def process_all_students(folder_path):
    students_results = []
    # Get all student image folders and sort them to maintain order
    # student_image_folders = sorted(os.listdir(base_folder_path))
    for student_folder in os.listdir(folder_path):
        student_folder_path = os.path.join(folder_path, student_folder)
        if os.path.isdir(student_folder_path):
            files = sorted(os.listdir(student_folder_path))
            first_column_path = second_column_path = student_id_path = None

            for file in files:
                if file.endswith('.jpg'):
                    if 'first_column' in file:
                        first_column_path = os.path.join(student_folder_path, file)
                    elif 'second_column' in file:
                        second_column_path = os.path.join(student_folder_path, file)
                    elif 'student_id' in file:
                        student_id_path = os.path.join(student_folder_path, file)

            if not all([first_column_path, second_column_path, student_id_path]):
                print(f"Missing ROI images in folder: {student_folder_path}")
                continue

            student_id = process_student_id(cv2.imread(student_id_path))
            responses_first = process_roi(cv2.imread(first_column_path), start_question_num=1)
            responses_second = process_roi(cv2.imread(second_column_path), start_question_num=26)

            student_data = {
                "studentID": student_id,
                "answers": {**responses_first, **responses_second}
            }
            students_results.append(student_data)

    final_output = {"students": students_results}

    with open('result_data.json', 'w') as json_file:
        json.dump(final_output, json_file, indent=4)

    print("Processing complete. Data saved to 'result_data.json'.")

# folder_path = 'data/ROIs'
# process_all_students(folder_path)


