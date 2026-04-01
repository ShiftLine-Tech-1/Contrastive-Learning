"""A text preprocessing module for cleaning and normalizing strings."""
from enums.cleaning_function import CleaningFunction


class Cleaner:
    """
    A text preprocessing pipeline that applies a sequence of cleaning methods.

    By default, it removes emails, usernames, links, and line breaks,
    normalizes dashes/quotes, and replaces dates/times with tokens.
    A custom pipeline can be defined by passing a list of method names.
    """

    def __init__(self, functions: list[CleaningFunction] | None = None) -> None:
        """
        Initializes the Cleaner.

        Args:
            functions: A list of method names (strings) to execute.
                If None, a default cleaning pipeline is used.

        """
        if functions is None:
            functions = [
                CleaningFunction.REMOVE_LINEBREAKS,
                CleaningFunction.REMOVE_AMP,
                CleaningFunction.NORMALIZE_DASHES,
                CleaningFunction.NORMALIZE_QUOTES,
                CleaningFunction.REPLACE_LINK,
                CleaningFunction.REPLACE_TELEGRAM_LINK,
                CleaningFunction.REPLACE_EMAIL,
                CleaningFunction.REPLACE_USERNAME,
                CleaningFunction.REPLACE_DATE,
                CleaningFunction.REPLACE_TIME,
                CleaningFunction.REMOVE_CHARS,
                CleaningFunction.REMOVE_SPACES,
            ]

        self._functions = functions

    def process(self, text: str) -> str:
        """Applies the configured cleaning pipeline to the text."""
        for cleaning_func in self._functions:
            text = cleaning_func(text)

        return text
