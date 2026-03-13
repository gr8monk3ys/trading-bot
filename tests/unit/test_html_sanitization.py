from data.data_fetchers.fed_speech_fetcher import FedSpeechFetcher
from data.data_fetchers.sec_edgar_fetcher import SECEdgarFetcher


def test_fed_speech_fetcher_strips_uppercase_script_tags():
    fetcher = FedSpeechFetcher(cache_dir=":memory:")
    html = "<SCRIPT>window.secret = 1;</SCRIPT><div>Speech text</div>"

    text = fetcher._extract_speech_text(html)

    assert "window.secret" not in text
    assert text == "Speech text"


def test_sec_edgar_fetcher_strips_script_tags_with_whitespace():
    fetcher = SECEdgarFetcher(cache_dir=":memory:")
    html = "<script>window.secret = 1;</script   ><p>Filing text</p>"

    text = fetcher._extract_text_from_html(html)

    assert "window.secret" not in text
    assert text == "Filing text"
