import fitz
import os


def extract_images_from_pdf(pdf_path, output_folder):
    # Creating output folder 
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the PDF file
    pdf_document = fitz.open(pdf_path)

    # Determining PDF name for creating a sub-folder
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Creating a sub-folder for the PDF
    pdf_folder = os.path.join(output_folder, pdf_name)
    os.makedirs(pdf_folder, exist_ok=True)

    # Iterating over pages
    for page_number, page in enumerate(pdf_document):
        page_number += 1

        # Naming for the image
        image_filename = f"Image_{page_number}.jpg"

        original_pix = page.get_pixmap(matrix=fitz.Identity, colorspace=fitz.csRGB, clip=None, annots=True)

        scale_x = 1540 / original_pix.width
        scale_y = 2000 / original_pix.height

        matrix = fitz.Matrix(scale_x, scale_y)

        pix = page.get_pixmap(matrix=matrix, dpi=300, colorspace=fitz.csRGB, clip=None, annots=True)

        # Saving the image
        image_path = os.path.join(pdf_folder, image_filename)
        pix.save(image_path)

    pdf_document.close()

#
# if __name__ == "__main__":
#     pdf_filename = 'Scans-4-2-24.pdf'
#     pdf_path = os.path.join(os.getcwd(), pdf_filename)
#     output_folder = os.path.join(os.getcwd(), "data")
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     extract_images_from_pdf(pdf_path, output_folder)
