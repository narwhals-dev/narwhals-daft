from __future__ import annotations

from typing import TYPE_CHECKING

import narwhals_daft

if TYPE_CHECKING:
    from narwhals.plugins import Plugin

    from narwhals_daft.dataframe import DaftLazyFrame

plugin: Plugin[DaftLazyFrame, DaftLazyFrame] = narwhals_daft
