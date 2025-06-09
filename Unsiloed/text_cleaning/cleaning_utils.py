import re
import unicodedata
import html

def normalize_unicode(text):
    return unicodedata.normalize('NFKC', text)

def replace_ligatures(text):
    ligatures = {'ﬁ': 'fi', 'ﬂ': 'fl', 'ﬃ': 'ffi'}
    for lig, rep in ligatures.items():
        text = text.replace(lig, rep)
    return text

def normalize_quotes(text):
    return text.replace('“', '"').replace('”', '"').replace("‘", "'").replace("’", "'")

def clean_bullets(text):
    bullet_patterns = [r'^\s*[-*•]\s+', r'^\s*\d+\.\s+']
    for pattern in bullet_patterns:
        text = re.sub(pattern, '', text, flags=re.MULTILINE)
    return text

def group_paragraphs(text):
    return re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

def normalize_whitespace(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def decode_mime(text):
    try:
        return html.unescape(text)
    except Exception:
        return text
