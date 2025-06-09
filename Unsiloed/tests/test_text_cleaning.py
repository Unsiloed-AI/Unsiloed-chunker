from Unsiloed.text_cleaning.cleaning_pipeline import TextCleaningPipeline

def test_cleaning_pipeline():
    text = "ﬁ Test “quoted”\nparagraph\n1. Numbered"
    cleaned = TextCleaningPipeline().clean(text)
    assert 'ﬁ' not in cleaned
    assert '“' not in cleaned
    assert '\n' not in cleaned or cleaned.count('\n') < 2
