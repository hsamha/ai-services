from enum import StrEnum


class SourceType(StrEnum):
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"

    CSV = "csv"
    JSON = "json"

    # Recognised as a file, but not a kind we can read.
    UNKNOWN = "unknown"
