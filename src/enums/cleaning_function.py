import re
from enum import Enum


def replace_email(text: str) -> str:
    """Replaces email addresses with a <EMAIL> token."""
    return re.sub(r"([a-zA-Z0-9._-]+@[a-zA-Z0-9._-]+\.[a-zA-Z]+)", "<EMAIL>", text)

def replace_username(text: str) -> str:
    """Replaces @usernames with a <USERNAME> token."""
    return re.sub(r"(?<![A-Za-z])@[A-Za-z0-9_]+", "<USERNAME>", text)

def replace_link(text: str) -> str:
    """Replace HTTP/HTTPS links and www addresses with a <URL> token."""
    return re.sub(r"(https?://[^\s)]+|www\.[^\s)]+)", "<URL>", text)

def replace_telegram_link(text: str) -> str:
    """Replace telegram links with a <URL> token."""
    return re.sub(r"(tgs?://[^\s)]+|www\.[^\s)]+)", "<URL>", text)

def replace_date(text: str) -> str:
    """Replaces dates (DD.MM.YYYY or YYYY.MM.DD) with a <DATE> token."""
    return re.sub(r"\b(?:\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})\b", "<DATE>", text)

def replace_time(text: str) -> str:
    """Replaces times (HH:MM) with a <TIME> token."""
    return re.sub(r"\b(?:(?:0?[1-9]|1[0-2]):[0-5][0-9]\s?[AaPp][Mm]|(?:[01]?[0-9]|2[0-3]):[0-5][0-9])\b", "<TIME>", text)

def remove_amp(text: str) -> str:
    """Removes &amp HTML entities."""
    return re.sub(r"&amp\S+", " ", text)

def remove_linebreaks(text: str) -> str:
    """Replaces carriage returns and newlines with spaces."""
    return re.sub(r"[\r\n]", " ", text)

def remove_spaces(text: str) -> str:
    """Removes extra consecutive spaces."""
    return " ".join(text.split())

def remove_chars(text: str) -> str:
    """Removes special characters, keeping alphanumeric and basic punctuation."""
    return re.sub(r'[^A-Za-zА-Яа-яЁё0-9€₽$!@#$%^&*()_\-+={}\[\]:;"\<>,.?/\\|~ ]+', "", text)  # noqa: RUF001

def normalize_dashes(text: str) -> str:
    """Converts various dash and hyphen characters to a standard '-'."""
    return re.sub(r"[-‑−–—―]", "-", text)  # noqa: RUF001

def normalize_quotes(text: str) -> str:
    """Converts various quote characters to a standard double quote."""
    return re.sub(r"[«»“”„‘’`']", '"', text)  # noqa: RUF001


class CleaningFunction(Enum):
    REMOVE_LINEBREAKS = remove_linebreaks
    REMOVE_AMP = remove_amp
    NORMALIZE_DASHES = normalize_dashes
    NORMALIZE_QUOTES = normalize_quotes
    REPLACE_LINK = replace_link
    REPLACE_TELEGRAM_LINK = replace_telegram_link
    REPLACE_EMAIL = replace_email
    REPLACE_USERNAME = replace_username
    REPLACE_DATE = replace_date
    REPLACE_TIME = replace_time
    REMOVE_CHARS = remove_chars
    REMOVE_SPACES = remove_spaces
