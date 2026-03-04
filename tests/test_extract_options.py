from parsantic.extract import ExtractOptions
from parsantic.extract.options import MediaOptions


def test_extract_options_media_defaults():
    options = ExtractOptions()

    assert options.media.pdf_mode == "auto"
    assert options.media.raster_dpi == 200
    assert options.media.max_image_dim == 2048
    assert options.media.page_strategy == "auto"


def test_media_options_accepts_native_pdf_mode_with_custom_dpi():
    options = MediaOptions(pdf_mode="native", raster_dpi=300)

    assert options.pdf_mode == "native"
    assert options.raster_dpi == 300


def test_extract_options_keeps_existing_constructor_behavior():
    options = ExtractOptions(passes=2, max_workers=4)

    assert options.passes == 2
    assert options.max_workers == 4
