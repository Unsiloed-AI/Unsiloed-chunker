from .cleaning_utils import (
    normalize_unicode, replace_ligatures, normalize_quotes,
    clean_bullets, group_paragraphs, normalize_whitespace, decode_mime
)

class TextCleaningPipeline:
    def __init__(self, config=None):
        default_config = {
            'normalize_unicode': True,
            'replace_ligatures': True,
            'normalize_quotes': True,
            'clean_bullets': True,
            'group_paragraphs': True,
            'normalize_whitespace': True,
            'decode_mime': True,
        }
        self.config = config or default_config

    def clean(self, text):
        if self.config.get('normalize_unicode'):
            text = normalize_unicode(text)
        if self.config.get('replace_ligatures'):
            text = replace_ligatures(text)
        if self.config.get('normalize_quotes'):
            text = normalize_quotes(text)
        if self.config.get('clean_bullets'):
            text = clean_bullets(text)
        if self.config.get('group_paragraphs'):
            text = group_paragraphs(text)
        if self.config.get('normalize_whitespace'):
            text = normalize_whitespace(text)
        if self.config.get('decode_mime'):
            text = decode_mime(text)
        return text
