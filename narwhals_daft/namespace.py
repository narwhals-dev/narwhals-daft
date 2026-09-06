from __future__ import annotations

import operator
import warnings
from functools import reduce
from typing import TYPE_CHECKING, Any

import daft
import daft.functions as F
from daft import Window
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, not_implemented
from narwhals.compliant import CompliantNamespace

from narwhals_daft.dataframe import DaftLazyFrame
from narwhals_daft.expr import DaftExpr
from narwhals_daft.selectors import DaftSelectorNamespace
from narwhals_daft.utils import lit, narwhals_to_native_dtype

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from daft import DataFrame, Expression
    from narwhals._utils import Version
    from narwhals.dtypes import DType
    from narwhals.typing import ConcatMethod, CorrelationMethod
    from typing_extensions import TypeIs

    from narwhals_daft.expr import WindowInputs


_F64 = daft.DataType.float64()
_STRING = daft.DataType.string()


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

    def concat_str(
        self, *exprs: DaftExpr, separator: str, ignore_nulls: bool
    ) -> DaftExpr:
        def func(cols: list[Expression]) -> Expression:
            result = F.concat_ws(separator, *(col.cast(_STRING) for col in cols))
            if ignore_nulls:
                # `concat_ws` yields null when every input is null, Polars "".
                return result.fill_null("")
            null_mask = reduce(operator.or_, (col.is_null() for col in cols))
            return F.when(~null_mask, result)

        return self._expr._from_elementwise_horizontal_op(func, *exprs)

    def _pairwise_variances_and_covariance(
        self,
        df: DaftLazyFrame,
        a: DaftExpr,
        b: DaftExpr,
        *,
        ddof: int,
        window: Window | None,
    ) -> tuple[Expression, Expression, Expression]:
        """Variances of `a` and `b`, and their covariance, over the rows where both are non-null.

        NOTE: Daft has no covariance or correlation aggregates and does not allow
        nesting aggregations (or window functions), which rules out a centred
        two-pass computation. The covariance is instead recovered from Daft's
        native variance kernel through the polarisation identity
        `2 * cov(a, b) = var(a + b) - var(a) - var(b)`, which holds for any common
        `ddof`. Daft's variance is itself a single-pass computation, so like
        `Expr.var`/`Expr.std` this loses precision when the mean is much larger
        than the spread (e.g. `var([1e9 + 1, 1e9 + 2, 1e9 + 3])` evaluates to 0).
        """
        a_ = df._evaluate_single_output_expr(a)
        b_ = df._evaluate_single_output_expr(b)
        valid = a_.not_null() & b_.not_null()
        a_ = F.when(valid, a_).otherwise(lit(None)).cast(_F64)
        b_ = F.when(valid, b_).otherwise(lit(None)).cast(_F64)
        var_a, var_b, var_ab = (F.var(expr, ddof) for expr in (a_, b_, a_ + b_))
        if window is not None:
            var_a, var_b, var_ab = (var.over(window) for var in (var_a, var_b, var_ab))
        return var_a, var_b, (var_ab - var_a - var_b) / lit(2.0)

    def _corr_cov(
        self,
        a: DaftExpr,
        b: DaftExpr,
        func: Callable[[DaftLazyFrame, Window | None], Expression],
    ) -> DaftExpr:
        def call(df: DaftLazyFrame) -> list[Expression]:
            return [func(df, None)]

        def window_function(
            df: DaftLazyFrame, inputs: WindowInputs
        ) -> list[Expression]:
            assert not inputs.order_by  # noqa: S101
            window = Window().partition_by(*(inputs.partition_by or [lit(1)]))
            return [func(df, window)]

        return self._expr(
            call,
            window_function,
            evaluate_output_names=combine_evaluate_output_names(a, b),
            alias_output_names=combine_alias_output_names(a, b),
            version=self._version,
        )

    def corr(self, a: DaftExpr, b: DaftExpr, *, method: CorrelationMethod) -> DaftExpr:
        if method != "pearson":
            msg = "Only 'pearson' correlation is supported for Daft."
            raise NotImplementedError(msg)

        def corr(df: DaftLazyFrame, window: Window | None) -> Expression:
            var_a, var_b, cov_ab = self._pairwise_variances_and_covariance(
                df, a, b, ddof=0, window=window
            )
            denominator = (var_a * var_b).sqrt()
            # The correlation is undefined when either variance is zero.
            return F.when(denominator > lit(0.0), cov_ab / denominator)

        return self._corr_cov(a, b, corr)

    def cov(self, a: DaftExpr, b: DaftExpr, *, ddof: int) -> DaftExpr:
        def cov(df: DaftLazyFrame, window: Window | None) -> Expression:
            _, _, cov_ab = self._pairwise_variances_and_covariance(
                df, a, b, ddof=ddof, window=window
            )
            # `var` is already null when there are no more than `ddof` valid pairs.
            return cov_ab

        return self._corr_cov(a, b, cov)

    struct = not_implemented()
