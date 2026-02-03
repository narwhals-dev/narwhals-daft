from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from narwhals.utils import Version
    from typing_extensions import TypeIs

    from narwhals_daft.dataframe import DaftLazyFrame
    from narwhals_daft.namespace import DaftNamespace


def __narwhals_namespace__(version: Version) -> DaftNamespace:  # noqa: N807
    from narwhals_daft.namespace import DaftNamespace

    return DaftNamespace(version=version)


def is_native(native_object: object) -> TypeIs[DaftLazyFrame]:
    import daft

    return isinstance(native_object, daft.DataFrame)


NATIVE_PACKAGE = "daft"
