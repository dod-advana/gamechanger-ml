from PyPDF2 import PdfFileReader


class PDFDocument:
    @staticmethod
    def bookmark_dict(reader, bookmark_list):
        """Get bookmark pages for a PDF.

        @remark Page numbers start at 0.

        Args:
            reader (Py2PDF.PdfFileReader)
            bookmark_list: reader.getOutlines() where reader is a 
                Py2PDF.PdfFileReader object.

        Returns: 
            dict: Dictionary such that keys are (int) page numbers and values 
                are (str) bookmark titles.
        """
        result = {}
        for item in bookmark_list:
            if isinstance(item, list):
                # recursive call
                result.update(PDFDocument.bookmark_dict(reader, item))
            else:
                result[reader.getDestinationPageNumber(item)] = item.title
        return result

    @staticmethod
    def num_pages(pdf_path):
        """Get the number of pages a pdf document has. 

        Args:
            pdf_path (str): Path to the pdf document.

        Returns:
            int: Number of pages.
        """
        reader = PdfFileReader(pdf_path)
        return reader.numPages

    @staticmethod
    def get_ref_section_page_nums(pdf_path):
        """Get the page numbers of a document's References section.

        Args:
            pdf_path (str): Path to the pdf file.
            
        Returns:
            int, int: The start and end page numbers.
        """
        reader = PdfFileReader(pdf_path)
        bookmarks = PDFDocument.bookmark_dict(reader, reader.getOutlines())
        num_refs = 0
        pages = list(bookmarks.keys())
        titles = list(bookmarks.values())
        start = None
        end = None

        for i in range(len(bookmarks)):
            if "references" in titles[i].lower():
                num_refs += 1
                start = pages[i]
                if i == len(bookmarks) - 1:
                    end = PDFDocument.num_pages(pdf_path) - 1
                else:
                    end = pages[i + 1] - 1

        if num_refs > 1:
            print(
                f"ERR: MORE THAN 1 REFERENCES BOOKMARK DETECTED for file: "
                f"{pdf_path}. Returning None"
            )
            return None, None

        if end and start and end < start:
            print(f"ERR: end < start. end: {end}, start: {start}")
            return None, None

        return start, end
