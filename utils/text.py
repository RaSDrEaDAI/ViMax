import re


def safe_path_component(name) -> str:
    """Sanitize an LLM-derived identifier for use as a filesystem path component.

    Identifiers come from model output over user-supplied story text, so they may
    contain separators or traversal sequences; keep word characters (including
    CJK), dashes, dots, spaces, parentheses and apostrophes, replace everything
    else, and strip leading dots so the result can never escape or hide within
    the working directory.

    Port of hkuds/vimax df1480d, with one deliberate widening: upstream also
    replaces ( ) and ', but recorded runs here have character dirs like
    "0_The Wednesday Chef (Marion)" — renaming those would orphan their existing
    portraits on resume and re-spend them. All three are legal path characters
    on Windows and POSIX and cannot form a separator or traversal.
    """
    cleaned = re.sub(r"[^\w\-. ()']", "_", str(name))
    cleaned = cleaned.strip().lstrip(".")
    return cleaned or "unnamed"
