from parsantic.extract import ExtractOptions
from parsantic.extract.options import MediaOptions


def test_extract_options_media_defaults():
    options = ExtractOptions()

    assert options.media.pdf_mode == "auto"
    assert options.media.raster_dpi == 200
    assert options.media.max_image_dim == 2048
    assert options.media.page_strategy == "auto"
    assert options.media.raster_format == "jpeg"
    assert options.media.jpeg_quality == 85
    assert options.repair == "targeted"
    assert options.max_repair_attempts == 2


def test_media_options_accepts_native_pdf_mode_with_custom_dpi():
    options = MediaOptions(pdf_mode="native", raster_dpi=300, raster_format="png", jpeg_quality=90)

    assert options.pdf_mode == "native"
    assert options.raster_dpi == 300
    assert options.raster_format == "png"
    assert options.jpeg_quality == 90


def test_extract_options_keeps_existing_constructor_behavior():
    options = ExtractOptions(passes=2, max_workers=4)

    assert options.passes == 2
    assert options.max_workers == 4
