from ..helpers import format_name


def send(message: str) -> str:
    payload = format_name(message)
    return f"sent:{payload}"

