from __future__ import annotations

import operator
import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any

import daft
import daft.functions as F
from narwhals._utils import Implementation, not_implemented
from narwhals.compliant import CompliantNamespace

from narwhals_daft.dataframe import DaftLazyFrame
from narwhals_daft.expr import DaftExpr
from narwhals_daft.selectors import DaftSelectorNamespace
from narwhals_daft.utils import lit, narwhals_to_native_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable

    from daft import DataFrame, Expression
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod
    from typing_extensions import TypeIs

    from narwhals_daft.expr import WindowInputs


class DaftNamespace(CompliantNamespace[DaftLazyFrame, DaftExpr]):
    _implementation: Implementation = Implementation.UNKNOWN

    def __init__(self, *, version: Version) -> None:
        self._version = version

    def is_native(self, native_object: object) -> TypeIs[DataFrame]:
        return isinstance(native_object, daft.DataFrame)

    def from_native(self, native_object: daft.DataFrame) -> DaftLazyFrame:
        return DaftLazyFrame(native_object, version=self._version)

    @property
    def selectors(self) -> DaftSelectorNamespace:
        return DaftSelectorNamespace.from_namespace(self)

    @property
    def _expr(self) -> type[DaftExpr]:
        return DaftExpr

    @property
    def _lazyframe(self) -> type[DaftLazyFrame]:
        return DaftLazyFrame

    def lit(self, value: Any, dtype: DType | type[DType] | None) -> DaftExpr:
        def func(_df: DaftLazyFrame) -> list[Expression]:
            if dtype is not None:
                return [lit(value).cast(narwhals_to_native_dtype(dtype, self._version))]
            return [lit(value)]

        def window_func(
            df: DaftLazyFrame, _window_inputs: WindowInputs
        ) -> list[Expression]:
            return func(df)

        return DaftExpr(
            func,
            window_func,
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            version=self._version,
        )

    def concat(
        self, items: Iterable[DaftLazyFrame], *, how: ConcatMethod
    ) -> DaftLazyFrame:
        list_items = list(items)
        native_items = (item._native_frame for item in items)
        if how == "diagonal":
            return DaftLazyFrame(
                reduce(lambda x, y: x.union_all_by_name(y), native_items),
                version=self._version,
            )
        first = list_items[0]
        schema = first.schema
        if how == "vertical" and not all(x.schema == schema for x in list_items[1:]):
            msg = "inputs should all have the same schema"
            raise TypeError(msg)
        res = reduce(lambda x, y: x.union(y), native_items)
        return first._with_native(res)

    def all_horizontal(self, *exprs: DaftExpr, ignore_nulls: bool) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            it = (F.coalesce(col, lit(True)) for col in cols) if ignore_nulls else cols
            return reduce(operator.and_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def any_horizontal(self, *exprs: DaftExpr, ignore_nulls: bool) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            it = (F.coalesce(col, lit(False)) for col in cols) if ignore_nulls else cols
            return reduce(operator.or_, it)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def sum_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*`daft\\.list_` is deprecated",
                    category=DeprecationWarning,
                )
                return daft.functions.columns_sum(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def max_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*`daft\\.list_` is deprecated",
                    category=DeprecationWarning,
                )
                return daft.functions.columns_max(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def min_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*`daft\\.list_` is deprecated",
                    category=DeprecationWarning,
                )
                return daft.functions.columns_min(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def mean_horizontal(self, *exprs: DaftExpr) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*`daft\\.list_` is deprecated",
                    category=DeprecationWarning,
                )
                return daft.functions.columns_mean(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def len(self) -> DaftExpr:
        def func(_df: DaftLazyFrame) -> list[Expression]:
            if not _df.columns:  # pragma: no cover
                msg = "Cannot use `nw.len()` on Daft DataFrame with zero columns"
                raise ValueError(msg)
            return [daft.col(_df.columns[0]).count(mode="all")]

        return DaftExpr(
            call=func,
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            version=self._version,
        )

    def when_then(
        self, predicate: DaftExpr, then: DaftExpr, otherwise: DaftExpr | None = None
    ) -> DaftExpr:
        def func(cols: list[Expression]) -> Expression:
            return F.when(cols[1], cols[0])

        def func_with_otherwise(cols: list[Expression]) -> Expression:
            return F.when(cols[1], cols[0]).otherwise(cols[2])

        if otherwise is None:
            return self._expr._from_elementwise_horizontal_op(func, then, predicate)
        return self._expr._from_elementwise_horizontal_op(
            func_with_otherwise, then, predicate, otherwise
        )

    def coalesce(self, *exprs: DaftExpr) -> DaftExpr:
        def func(cols: Iterable[Expression]) -> Expression:
            return daft.functions.coalesce(*cols)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    concat_str = not_implemented()
    corr = not_implemented()
    cov = not_implemented()

    def struct(self, *exprs: DaftExpr) -> DaftExpr:
        from narwhals._expression_parsing import (
            combine_alias_output_names,
            combine_evaluate_output_names,
            evaluate_output_names_and_aliases,
        )

        def func(df: DaftLazyFrame) -> list[Expression]:
            import daft.functions.struct as _struct

            names_to_cols = {
                alias: native_expr
                for expr in exprs
                for native_expr, _, alias in zip(
                    expr(df),
                    *evaluate_output_names_and_aliases(expr, df, []),
                    strict=True,
                )
            }
            aliased = [col.alias(name) for name, col in names_to_cols.items()]
            return [_struct.to_struct(*aliased)]

        return self._expr(
            func,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            version=self._version,
        )
