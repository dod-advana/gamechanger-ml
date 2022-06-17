# gamechangerml/src/section_classifier

The Section Classifier model is used to parse documents into sections and assign labels to those sections.

## Structure

```
gamechangerml/src/section_classifier/
├──section_classifier.py                    Section classifier model
├──document_section.py                      Document section definition
├──document_sections.py                     Parse and label document sections
├──configs/
│  ├──__init__.py
│  ├──model_configs.py                      Section classifier model configs
│  └──fields.py                             Shared field names
├──test/                                    Tests
│  ├──__init__.py
│  ├──test_document_section.py
│  ├──test_document_sections.py
│  └──test_section_classifier.py
```

## Example Usage

```python
from gamechangerml.src.section_classifier import SectionClassifier

# Get a document's References section.
classifier = SectionClassifier()
# Assume doc_dict is already defined as a dict representation of a document.
sections = DocumentSections(doc_dict, classifier.tokenizer, classifier.pipeline)
references_section = sections.references_section  # Ta da!
```

## Tests

- See [here](test/README.md)
