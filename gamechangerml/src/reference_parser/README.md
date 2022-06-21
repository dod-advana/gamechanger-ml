# gamechangerml/src/reference_parser

The Reference Parser is used to split a document's References section into individual references.

## Structure

```
gamechangerml/src/reference_parser/
├──__init__.py
├──abc/                                     Parse References sections that have (a), (b), (c),... format
│  ├──__init__.py
│  ├──split_refs_abc.py
│  └──utils/
│     ├──__init__.py
│     ├──rm_see_enclosure.py
│     └──split_by_date_and_kws.py
├──non_abc/                                 Parse References sections that do not have (a), (b), (c)... format
│  ├──__init__.py
│  ├──split_refs_non_abc.py
│  └──utils/
│     ├──__init__.py
│     ├──clean_ref_line.py
│     ├──filter_refs.py
│     └──split_by_date_and_kws.py
├──pdf_to_docx/                             Convert pdf documents to docx
│  ├──__init__.py
│  ├──docx_document.py
│  ├──pdf_document.py
│  └──pdf_to_docx.py
├──shared/
│  ├──__init__.py
│  ├──is_abc_format.py
│  └──ref_end_patterns.py
├──test/
│  ├──__init__.py
│  ├──README.md
│  ├──test_abc.py
│  ├──test_abc.py
│  └──test_shared.py
```

## Example Usage

```python
from gamechangerml.src.section_classifier import SectionClassifier, DocumentSections
from gamechangerml.src.reference_parser.shared import is_abc_format
from gamechangerml.src.reference_parser import split_refs_abc, split_refs_non_abc

classifier = SectionClassifier()
# Assume doc_dict is already defined as a dict representation of a document.
sections = DocumentSections(doc_dict, classifier.tokenizer, classifier.pipeline)
ref_section = sections.references_section

if is_abc_format(ref_section):
    parsed_refs = split_refs_abc(ref_section)
else:
    # Assume pdf_path is already defined as a str path to the source pdf.
    parsed_refs = split_refs_non_abc(
        pdf_path,
        ref_section,
        classifier.tokenizer,
        classifier.pipe
    )
```

## Tests

- See [here](test/README.md)
