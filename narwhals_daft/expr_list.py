from __future__ import annotations

from typing import TYPE_CHECKING, Any

import daft.functions as F
from daft import DataType, lit
from narwhals.compliant import ListNamespace

if TYPE_CHECKING:
    from daft import Expression
    from narwhals.typing import NonNestedLiteral

    from narwhals_daft.dataframe import DaftLazyFrame
    from narwhals_daft.expr import DaftExpr


class ExprListNamespace(ListNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr, /) -> None:
        self._compliant = expr

    @property
    def compliant(self) -> DaftExpr:
        return self._compliant

    def len(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.list_count(expr, "all"))

    def min(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.list_min)

    def max(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.list_max)

    def mean(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.list_mean)

    def sum(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            return F.when(F.list_count(expr, "valid") == lit(0), lit(0)).otherwise(
                F.list_sum(expr)
            )

        return self.compliant._with_elementwise(func)

    def sort(self, *, descending: bool, nulls_last: bool) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.list_sort(expr, desc=descending, nulls_first=not nulls_last)
        )

    def unique(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            expr_distinct = F.list_distinct(expr)
            return F.when(
                F.list_count(expr, "null") == lit(0), expr_distinct
            ).otherwise(F.list_append(expr_distinct, lit(None)))

        return self.compliant._with_elementwise(func)

    def get(self, index: int) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.get(expr, key=index))

    def median(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            sorted_expr = F.list_sort(expr, nulls_first=False)
            size = F.list_count(sorted_expr, mode="valid")
            mid_index = (size / lit(2)).cast("int").fill_null(0)
            odd_case = F.get(sorted_expr, key=mid_index)
            even_case = (
                F.get(sorted_expr, key=mid_index - lit(1))
                + F.get(sorted_expr, key=mid_index)
            ) / lit(2)
            return (
                F.when((size.is_null()) | (size == lit(0)), lit(None))
                .when(size % lit(2) == lit(1), odd_case)
                .otherwise(even_case)
            )

        return self.compliant._with_elementwise(func)

    def contains(self, item: NonNestedLiteral) -> DaftExpr:
        if item is None:
            return self.compliant._with_elementwise(
                lambda expr: F.list_count(expr, "null") > lit(0)
            )
        item_dtype = DataType.infer_from_object(item)

        def func(df: DaftLazyFrame, expr: Expression) -> Expression:
            # `list_contains` requires `item` to have exactly the element type of
            # the list, while a Python literal only ever infers to Int64, Float64,
            # Timestamp[us], ... so cast it, when that is lossless, to the element
            # type resolved against the frame's schema (plan-time only).
            list_dtype = df._native_dtype(expr)
            if not (list_dtype.is_list() or list_dtype.is_fixed_size_list()):
                return F.list_contains(expr, lit(item))
            element_dtype = list_dtype.dtype
            if not _is_same_family(element_dtype, item_dtype):
                return F.list_contains(expr, lit(item))
            if not _is_representable(item, element_dtype):
                # e.g. `1.5` or `300` can never be an element of an Int8 list.
                return F.when(expr.not_null(), lit(False))
            return F.list_contains(expr, lit(item).cast(element_dtype))

        return self.compliant._with_elementwise_frame(func)


_INTEGER_BOUNDS: dict[str, tuple[int, int]] = {
    "Int8": (-(1 << 7), (1 << 7) - 1),
    "Int16": (-(1 << 15), (1 << 15) - 1),
    "Int32": (-(1 << 31), (1 << 31) - 1),
    "Int64": (-(1 << 63), (1 << 63) - 1),
    "UInt8": (0, (1 << 8) - 1),
    "UInt16": (0, (1 << 16) - 1),
    "UInt32": (0, (1 << 32) - 1),
    "UInt64": (0, (1 << 64) - 1),
}


def _is_same_family(element_dtype: DataType, item_dtype: DataType) -> bool:
    """Whether casting a literal of `item_dtype` to `element_dtype` is meaningful."""
    return (
        (element_dtype.is_numeric() and item_dtype.is_numeric())
        or (element_dtype.is_timestamp() and item_dtype.is_timestamp())
        or (element_dtype.is_duration() and item_dtype.is_duration())
        or (element_dtype.is_time() and item_dtype.is_time())
    )


def _is_representable(item: Any, element_dtype: DataType) -> bool:
    """Whether the numeric `item` survives a cast to an integer `element_dtype`."""
    if not element_dtype.is_integer():
        return True
    lower, upper = _INTEGER_BOUNDS[str(element_dtype)]
    is_integral = isinstance(item, int) or item.is_integer()
    return is_integral and lower <= item <= upper
