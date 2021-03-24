import warnings

__all__ = [
    'silence_pydicom_warnings',
]


def silence_pydicom_warnings() -> None:
    warnings.filterwarnings('ignore', category=UserWarning, module='pydicom.pixel_data_handlers.pillow_handler')
