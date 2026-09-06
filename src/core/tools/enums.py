from enum import StrEnum


class FileType(StrEnum):
    """What kind of document a file holds. What decides which loader reads it."""

    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"

    CSV = "csv"
    JSON = "json"

    UNKNOWN = "unknown"
