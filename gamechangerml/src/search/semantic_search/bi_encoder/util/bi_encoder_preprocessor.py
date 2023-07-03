from re import sub


class BiEncoderPreprocessor:
    @staticmethod
    def process(text: str, remove_newlines: bool = False) -> str:
        # Fix spacing around punctuation.
        rm_space_before_marks = """)-!.,’"':?"""
        for mark in rm_space_before_marks:
            text = text.replace(" " + mark, mark)

        rm_space_after_marks = """(-'"""
        for mark in rm_space_after_marks:
            text = text.replace(mark + " ", mark)

        if remove_newlines:
            text = " ".join(text.split()).strip()
        else:
            text = sub(r" {2,}", " ", text).strip()

        return text
