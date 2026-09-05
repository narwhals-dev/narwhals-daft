from __future__ import annotations

from typing import TYPE_CHECKING, cast

import daft
import daft.exceptions
from daft import DataType
from narwhals._utils import isinstance_or_issubclass
from narwhals.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    NarwhalsError,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit

    from narwhals_daft.dataframe import DaftLazyFrame
    from narwhals_daft.expr import DaftExpr

lit = daft.lit
"""Alias for `daft.lit`."""


def evaluate_exprs(
    df: DaftLazyFrame, /, *exprs: DaftExpr
) -> list[tuple[str, daft.Expression]]:
    native_results: list[tuple[str, daft.Expression]] = []
    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.extend(zip(output_names, native_series_list, strict=True))
    return native_results


def native_to_narwhals_dtype(daft_dtype: DataType, version: Version) -> DType:  # noqa: PLR0912,C901,PLR0911
    dtypes = version.dtypes

    if daft_dtype == DataType.int64():
        return dtypes.Int64()
    if daft_dtype == DataType.int32():
        return dtypes.Int32()
    if daft_dtype == DataType.int16():
        return dtypes.Int16()
    if daft_dtype == DataType.int8():
        return dtypes.Int8()
    if daft_dtype == DataType.uint64():
        return dtypes.UInt64()
    if daft_dtype == DataType.uint32():
        return dtypes.UInt32()
    if daft_dtype == DataType.uint16():
        return dtypes.UInt16()
    if daft_dtype == DataType.uint8():
        return dtypes.UInt8()
    if daft_dtype == DataType.float64():
        return dtypes.Float64()
    if daft_dtype == DataType.float32():
        return dtypes.Float32()
    if daft_dtype == DataType.string():
        return dtypes.String()
    if daft_dtype == DataType.date():
        return dtypes.Date()
    if daft_dtype.is_timestamp():
        time_unit = cast("TimeUnit", str(daft_dtype.timeunit))
        return dtypes.Datetime(time_unit, daft_dtype.timezone)
    if daft_dtype == DataType.bool():
        return dtypes.Boolean()
    if daft_dtype.is_duration():
        return dtypes.Duration(cast("TimeUnit", str(daft_dtype.timeunit)))
    if daft_dtype.is_time():
        return dtypes.Time()
    if daft_dtype == DataType.decimal128(1, 1):  # pragma: no cover
        return dtypes.Decimal()
    if DataType.is_fixed_size_list(daft_dtype):
        return dtypes.Array(
            native_to_narwhals_dtype(daft_dtype.dtype, version), daft_dtype.size
        )
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(  # noqa: PLR0912,C901,PLR0911
    dtype: DType | type[DType], version: Version
) -> daft.DataType:
    dtypes = version.dtypes
    if dtype == dtypes.Float64:
        return DataType.float64()
    if dtype == dtypes.Float32:
        return DataType.float32()
    if dtype in {dtypes.Int128, dtypes.UInt128}:  # pragma: no cover
        msg = "Converting to Int128/UInt128 is not (yet) supported for Daft."
        raise NotImplementedError(msg)
    if dtype == dtypes.Int64:
        return DataType.int64()
    if dtype == dtypes.Int32:
        return DataType.int32()
    if dtype == dtypes.Int16:
        return DataType.int16()
    if dtype == dtypes.Int8:
        return DataType.int8()
    if dtype == dtypes.UInt64:
        return DataType.uint64()
    if dtype == dtypes.UInt32:
        return DataType.uint32()
    if dtype == dtypes.UInt16:
        return DataType.uint16()
    if dtype == dtypes.UInt8:
        return DataType.uint8()
    if dtype == dtypes.String:
        return DataType.string()
    if dtype == dtypes.Boolean:
        return DataType.bool()
    if dtype == dtypes.Object:  # pragma: no cover
        msg = "Converting to Object is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Categorical:
        msg = "Converting to Categorical is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Enum:
        msg = "Converting to Enum is not (yet) supported for Daft"
        raise NotImplementedError(msg)
    if dtype == dtypes.Date:
        return DataType.date()
    if dtype == dtypes.Time:
        return DataType.time("ns")
    if dtype == dtypes.Binary:
        return DataType.binary()
    if dtype == dtypes.Decimal:  # pragma: no cover
        msg = "Casting to Decimal is not supported yet."
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        return DataType.timestamp(dtype.time_unit, dtype.time_zone)
    if isinstance_or_issubclass(dtype, dtypes.Duration):
        return DataType.duration(dtype.time_unit)
    if isinstance_or_issubclass(dtype, dtypes.List):
        return DataType.list(narwhals_to_native_dtype(dtype.inner, version))
    if isinstance_or_issubclass(dtype, dtypes.Struct):
        return DataType.struct(
            {
                field.name: narwhals_to_native_dtype(field.dtype, version)
                for field in dtype.fields
            }
        )
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        return DataType.fixed_size_list(
            narwhals_to_native_dtype(dtype.inner, version), dtype.size
        )
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def extend_bool(
    value: bool | Iterable[bool],  # noqa: FBT001
    n_match: int,
) -> Sequence[bool]:
    """Ensure the given bool or sequence of bools is the correct length.

    Stolen from https://github.com/pola-rs/polars/blob/b8bfb07a4a37a8d449d6d1841e345817431142df/py-polars/polars/_utils/various.py#L580-L594
    """
    return (value,) * n_match if isinstance(value, bool) else tuple(value)


def evaluate_literal(expr: DaftExpr) -> object:
    return next(expr._metadata.op_nodes_reversed()).kwargs["value"]


def catch_daft_exception(
    exception: Exception, frame: DaftLazyFrame, /
) -> NarwhalsError | Exception:
    """Translate a Daft exception into its narwhals counterpart, keeping the message."""
    if not isinstance(exception, daft.exceptions.DaftCoreException):
        return exception  # pragma: no cover
    msg = str(exception)
    if "not found" in msg:
        return ColumnNotFoundError(
            f"{msg}\n\nHint: Did you mean one of these columns: {frame.columns}?"
        )
    if "duplicate" in msg.lower():
        return DuplicateError(msg)
    if "TypeError" in msg:
        return InvalidOperationError(msg)
    if "ComputeError" in msg:
        return ComputeError(msg)
    return NarwhalsError(msg)
