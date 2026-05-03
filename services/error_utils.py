# services/error_utils.py
from __future__ import annotations

import logging
import traceback
from typing import Callable, Any

import streamlit as st


logger = logging.getLogger(__name__)


def show_error(message: str, error: Exception | None = None, show_details: bool = False) -> None:
    st.error(message)

    if error is not None:
        logger.exception(message)

        if show_details:
            with st.expander("Tekniset tiedot"):
                st.code("".join(traceback.format_exception(type(error), error, error.__traceback__)))


def safe_render(section_name: str, render_func: Callable[[], Any], show_details: bool = False) -> None:
    try:
        render_func()
    except Exception as e:
        show_error(
            f"{section_name}-osion lataus epäonnistui. Yritä päivittää sivu hetken päästä.",
            error=e,
            show_details=show_details,
        )


def safe_value(section_name: str, func: Callable[[], Any], fallback: Any = None, show_details: bool = False) -> Any:
    try:
        return func()
    except Exception as e:
        show_error(
            f"{section_name}-datan lataus epäonnistui.",
            error=e,
            show_details=show_details,
        )
        return fallback