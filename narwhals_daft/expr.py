from __future__ import annotations

import operator as op
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

import daft.functions as F
from daft import Window, col, lit
from narwhals._compliant import CompliantExpr
from narwhals._expression_parsing import (
    combine_alias_output_names,
    combine_evaluate_output_names,
)
from narwhals._utils import Implementation, not_implemented

from narwhals_daft.expr_name import ExprNameNamespace
from narwhals_daft.expr_str import ExprStringNamespace
from narwhals_daft.utils import evaluate_literal, extend_bool, narwhals_to_native_dtype

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from daft import Expression
    from narwhals._compliant.typing import AliasNames, EvalNames, EvalSeries
    from narwhals._utils import Version, _LimitedContext
    from narwhals.dtypes import DType
    from typing_extensions import Self, TypeIs

    from narwhals_daft.dataframe import DaftLazyFrame
    from narwhals_daft.namespace import DaftNamespace

    WindowFunction: TypeAlias = Callable[
        [DaftLazyFrame, "WindowInputs"], Sequence[Expression]
    ]


class WindowInputs:
    __slots__ = ("order_by", "partition_by")

    def __init__(
        self, partition_by: Sequence[str | Expression], order_by: Sequence[str]
    ) -> None:
        self.partition_by = partition_by
        self.order_by = order_by


class DaftExpr(CompliantExpr["DaftLazyFrame", "Expression"]):
    _implementation = Implementation.UNKNOWN

    def __init__(
        self,
        call: Callable[[DaftLazyFrame], Sequence[Expression]],
        window_function: WindowFunction | None = None,
        *,
        evaluate_output_names: EvalNames[DaftLazyFrame],
        alias_output_names: AliasNames | None,
        version: Version,
    ) -> None:
        self._call = call
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._version = version
        self._window_function: WindowFunction | None = window_function

    def _partition_by(self, *cols: Expression | str) -> Window:
        """Wraps `Window().partitionBy`, with default and `WindowInputs` handling."""
        return Window().partition_by(*cols or [lit(1)])

    def _window_expression(
        self,
        expr: Expression,
        partition_by: Sequence[str | Expression] = (),
        order_by: Sequence[str | Expression] = (),
        rows_start: int | None = None,
        rows_end: int | None = None,
        *,
        descending: list[bool] | None = None,
        nulls_first: list[bool] | None = None,
    ) -> Expression:
        window = self._partition_by(*partition_by)
        if order_by:
            window = window.order_by(
                *order_by,
                desc=descending or [False] * len(order_by),
                nulls_first=nulls_first or [True] * len(order_by),
            )
        if rows_start is not None and rows_end is not None:
            window = window.rows_between(rows_start, rows_end)
        elif rows_end is not None:
            window = window.rows_between(Window.unbounded_preceding, rows_end)
        elif rows_start is not None:  # pragma: no cover
            window = window.rows_between(rows_start, Window.unbounded_following)
        return expr.over(window)

    def _is_multi_output_unnamed(self) -> bool:
        """Return `True` for multi-output aggregations without names.

        For example, column `'a'` only appears in the output as a grouping key:

            df.group_by('a').agg(nw.all().sum())

        It does not get included in:

            nw.all().sum().
        """
        return self._metadata.expansion_kind.is_multi_unnamed()

    @property
    def window_function(self) -> WindowFunction:
        def default_window_func(
            df: DaftLazyFrame, inputs: WindowInputs
        ) -> Sequence[Expression]:
            assert not inputs.order_by  # noqa: S101
            return [
                self._window_expression(expr, inputs.partition_by) for expr in self(df)
            ]

        return self._window_function or default_window_func

    def _cum_window_func(
        self, func_name: Literal["sum", "max", "min", "product"], *, reverse: bool
    ) -> WindowFunction:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            descending = list(extend_bool(reverse, len(inputs.order_by)))
            nulls_first = list(extend_bool(not reverse, len(inputs.order_by)))
            return [
                F.when(
                    ~F.is_null(expr),
                    self._window_expression(
                        getattr(F, func_name)(expr),
                        inputs.partition_by,
                        inputs.order_by,
                        descending=descending,
                        nulls_first=nulls_first,
                        rows_end=0,
                    ),
                )
                for expr in self(df)
            ]

        return func

    def _rolling_window_func(
        self,
        func_name: Literal["sum", "mean", "std", "var"],
        window_size: int,
        min_samples: int,
        ddof: int | None = None,
        *,
        center: bool,
    ) -> WindowFunction:
        supported_funcs = ["sum", "mean", "std", "var"]
        if center:
            half = (window_size - 1) // 2
            remainder = (window_size - 1) % 2
            start = -(half + remainder)
            end = half
        else:
            start = -(window_size - 1)
            end = 0

        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            if func_name in {"sum", "mean"}:
                func_: str = func_name
            elif func_name == "var" and ddof == 0:
                func_ = "var_pop"
            elif func_name in "var" and ddof == 1:
                func_ = "var_samp"
            elif func_name == "std" and ddof == 0:
                func_ = "stddev_pop"
            elif func_name == "std" and ddof == 1:
                func_ = "stddev_samp"
            elif func_name in {"var", "std"}:  # pragma: no cover
                msg = f"Only ddof=0 and ddof=1 are currently supported for rolling_{func_name}."
                raise ValueError(msg)
            else:  # pragma: no cover
                msg = f"Only the following functions are supported: {supported_funcs}.\nGot: {func_name}."
                raise ValueError(msg)
            window_kwargs: Any = {
                "partition_by": inputs.partition_by,
                "order_by": inputs.order_by,
                "rows_start": start,
                "rows_end": end,
            }
            return [
                F.when(
                    self._window_expression(expr.count(), **window_kwargs)
                    >= lit(min_samples),
                    self._window_expression(getattr(F, func_)(expr), **window_kwargs),
                )
                for expr in self(df)
            ]

        return func

    def broadcast(self) -> Self:
        return self.over([lit(1)], [])

    def __call__(self, df: DaftLazyFrame) -> Sequence[Expression]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaftNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals_daft.namespace import DaftNamespace

        return DaftNamespace(version=self._version)

    @classmethod
    def _alias_native(cls, expr: Expression, name: str) -> Expression:
        return expr.alias(name)

    def alias(self, name: str) -> Self:
        def fn(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = (
                    f"Expected function with single output, found output names: {names}"
                )
                raise ValueError(msg)
            return [name]

        return self._with_alias_output_names(fn)

    @classmethod
    def from_column_names(
        cls: type[Self],
        evaluate_column_names: EvalNames[DaftLazyFrame],
        /,
        *,
        context: _LimitedContext,
    ) -> Self:
        def func(df: DaftLazyFrame) -> list[Expression]:
            return [col(col_name) for col_name in evaluate_column_names(df)]

        return cls(
            func,
            evaluate_output_names=evaluate_column_names,
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self], *column_indices: int, context: _LimitedContext
    ) -> Self:
        def func(df: DaftLazyFrame) -> list[Expression]:
            columns = df.columns
            return [col(columns[i]) for i in column_indices]

        return cls(
            func,
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            version=context._version,
        )

    @classmethod
    def _from_elementwise_horizontal_op(
        cls, func: Callable[[list[Expression]], Expression], *exprs: DaftExpr
    ) -> DaftExpr:
        def call(df: DaftLazyFrame) -> Sequence[Expression]:
            return [func([e for expr in exprs for e in expr(df)])]

        def window_function(
            df: DaftLazyFrame, window_inputs: WindowInputs
        ) -> Sequence[Expression]:
            lst = [e for expr in exprs for e in expr.window_function(df, window_inputs)]
            return [func(lst)]

        context = exprs[0]
        return cls(
            call,
            window_function=window_function,
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            version=context._version,
        )

    def _callable_to_eval_series(
        self, call: Callable[..., Expression], /, **expressifiable_args: Self | Any
    ) -> EvalSeries[DaftLazyFrame, Expression]:
        def func(df: DaftLazyFrame) -> list[Expression]:
            native_series_list = self(df)
            other_native_series = {
                key: df._evaluate_expr(value)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return func

    def _push_down_window_function(
        self, call: Callable[..., Expression], /, **expressifiable_args: DaftExpr
    ) -> WindowFunction:
        def window_f(
            df: DaftLazyFrame, window_inputs: WindowInputs
        ) -> Sequence[Expression]:
            # If a function `f` is elementwise, and `g` is another function, then
            # - `f(g) over (window)`
            # - `f(g over (window))
            # are equivalent.
            # Make sure to only use with if `call` is elementwise!
            native_series_list = self.window_function(df, window_inputs)
            other_native_series = {
                key: df._evaluate_window_expr(value, window_inputs)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return window_f

    def _with_callable(
        self,
        call: Callable[..., Expression],
        window_func: WindowFunction | None = None,
        /,
        **expressifiable_args: Self | Any,
    ) -> Self:
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            window_func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _with_elementwise(
        self, call: Callable[..., Expression], /, **expressifiable_args: DaftExpr
    ) -> Self:
        return self.__class__(
            self._callable_to_eval_series(call, **expressifiable_args),
            self._push_down_window_function(call, **expressifiable_args),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _with_binary(self, op: Callable[..., Expression], other: Self | Any) -> Self:
        return self.__class__(
            self._callable_to_eval_series(op, other=other),
            self._push_down_window_function(op, other=other),
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def _with_alias_output_names(self, func: AliasNames | None, /) -> Self:
        current_alias_output_names = self._alias_output_names
        alias_output_names = (
            None
            if func is None
            else func
            if current_alias_output_names is None
            else lambda output_names: func(current_alias_output_names(output_names))
        )
        return type(self)(
            self._call,
            self._window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            version=self._version,
        )

    def __and__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr & other), other=other)

    def __or__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr | other), other=other)

    def __invert__(self) -> Self:
        invert = cast("Callable[..., Expression]", op.invert)
        return self._with_elementwise(invert)

    def __add__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr + other), other)

    def __sub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr - other), other)

    def __rsub__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (other - expr), other).alias(
            "literal"
        )

    def __mul__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr * other), other)

    def __truediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr / other), other)

    def __rtruediv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (other / expr), other).alias(
            "literal"
        )

    def __floordiv__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr / other).floor(), other)

    def __rfloordiv__(self, other: Self) -> Self:
        return self._with_binary(
            lambda expr, other: (other / expr).floor(), other
        ).alias("literal")

    def __mod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr % other), other)

    def __rmod__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (other % expr), other).alias(
            "literal"
        )

    def __pow__(self, other: Self) -> Self:
        if other._metadata.is_literal:
            other_lit = evaluate_literal(other)
            return self._with_callable(lambda expr: (expr**other_lit))
        msg = "`pow` with non-literal input is not yet supported"
        raise NotImplementedError(msg)

    def __rpow__(self, other: Self) -> Self:
        if other._metadata.is_literal:
            other_lit = evaluate_literal(other)
            return self._with_callable(lambda expr: (other_lit**expr)).alias("literal")
        msg = "`__rpow__` with non-literal input is not yet supported"
        raise NotImplementedError(msg)

    def __gt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr > other), other)

    def __ge__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr >= other), other)

    def __lt__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr < other), other)

    def __le__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr <= other), other)

    def __eq__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr == other), other)

    def __ne__(self, other: Self) -> Self:
        return self._with_binary(lambda expr, other: (expr != other), other)

    def over(
        self, partition_by: Sequence[str | Expression], order_by: Sequence[str]
    ) -> Self:
        def func(df: DaftLazyFrame) -> Sequence[Expression]:
            return self.window_function(df, WindowInputs(partition_by, order_by))

        return self.__class__(
            func,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def all(self) -> Self:
        def f(expr: Expression) -> Expression:
            return F.coalesce(expr.bool_and(), lit(True))

        return self._with_callable(f)

    def any(self) -> Self:
        def f(expr: Expression) -> Expression:
            return F.coalesce(expr.bool_or(), lit(False))

        return self._with_callable(f)

    def cast(self, dtype: DType | type[DType]) -> Self:
        def func(expr: Expression) -> Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return expr.cast(native_dtype)

        def window_func(df: DaftLazyFrame, inputs: WindowInputs) -> list[Expression]:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return [
                expr.cast(native_dtype) for expr in self.window_function(df, inputs)
            ]

        return self._with_callable(func, window_func)

    def count(self) -> Self:
        return self._with_elementwise(lambda _input: _input.count("valid"))

    def abs(self) -> Self:
        return self._with_elementwise(lambda _input: _input.abs())

    def mean(self) -> Self:
        return self._with_callable(lambda _input: _input.mean())

    def clip(self, lower_bound: DaftExpr, upper_bound: DaftExpr) -> Self:
        return self._with_elementwise(
            lambda expr, lower_bound, upper_bound: expr.clip(lower_bound, upper_bound),
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )

    def clip_lower(self, lower_bound: DaftExpr) -> Self:
        return self._with_elementwise(
            lambda expr, lower_bound: expr.clip(lower_bound), lower_bound=lower_bound
        )

    def clip_upper(self, upper_bound: DaftExpr) -> Self:
        return self._with_elementwise(
            lambda expr, upper_bound: expr.clip(max=upper_bound),
            upper_bound=upper_bound,
        )

    def sum(self) -> Self:
        def f(expr: Expression) -> Expression:
            return F.coalesce(expr.sum(), lit(0))

        return self._with_callable(f)

    def n_unique(self) -> Self:
        return self._with_callable(
            lambda _input: _input.count_distinct() + _input.is_null().bool_or()
        )

    def len(self) -> Self:
        return self._with_callable(lambda _input: _input.count("all"))

    def std(self, ddof: int) -> Self:
        def func(expr: Expression) -> Expression:
            std_pop = expr.stddev()
            if ddof == 0:
                return std_pop
            n_samples = expr.count(mode="valid")
            return std_pop * n_samples.sqrt() / (n_samples - ddof).sqrt()

        def window_func(df: DaftLazyFrame, inputs: WindowInputs) -> list[Expression]:
            assert not inputs.order_by  # noqa: S101
            w = Window().partition_by(*inputs.partition_by)
            return [
                expr.stddev().over(w)
                * expr.count().over(w).sqrt()
                / (expr.count().over(w) - lit(ddof)).sqrt()
                for expr in self(df)
            ]

        return self._with_callable(func, window_func)

    def var(self, ddof: int) -> Self:
        def func(expr: Expression) -> Expression:
            std_pop = expr.stddev()
            var_pop = std_pop * std_pop
            if ddof == 0:
                return var_pop
            n_samples = expr.count(mode="valid")
            return var_pop * n_samples / (n_samples - ddof)

        return self._with_callable(func)

    def max(self) -> Self:
        return self._with_callable(lambda _input: _input.max())

    def min(self) -> Self:
        return self._with_callable(lambda _input: _input.min())

    def null_count(self) -> Self:
        return self._with_callable(lambda _input: _input.is_null().cast("uint32").sum())

    def is_null(self) -> Self:
        return self._with_elementwise(lambda _input: _input.is_null())

    def is_nan(self) -> Self:
        return self._with_elementwise(lambda _input: _input.float.is_nan())

    def is_finite(self) -> Self:
        return self._with_elementwise(
            lambda _input: (_input > float("-inf")) & (_input < float("inf"))
        )

    def is_in(self, other: Sequence[Any]) -> Self:
        return self._with_elementwise(lambda _input: _input.is_in(other))

    def round(self, decimals: int) -> Self:
        return self._with_elementwise(lambda _input: _input.round(decimals))

    def floor(self) -> Self:
        return self._with_elementwise(lambda _input: _input.floor())

    def ceil(self) -> Self:
        return self._with_elementwise(lambda _input: _input.ceil())

    def fill_null(self, value: Self | Any, strategy: Any, limit: int | None) -> Self:
        if strategy is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        return self._with_elementwise(
            lambda _input, value: _input.fill_null(value), value=value
        )

    def log(self, base: float) -> Self:
        return self._with_elementwise(lambda expr: expr.log(base=base))

    def exp(self) -> Self:
        return self._with_elementwise(lambda expr: expr.exp())

    def sqrt(self) -> Self:
        return self._with_elementwise(lambda expr: expr.sqrt())

    def skew(self) -> Self:
        return self._with_callable(lambda expr: expr.skew())

    @classmethod
    def _is_expr(cls, obj: Self | Any) -> TypeIs[Self]:
        return hasattr(obj, "__narwhals_expr__")

    def _with_window_function(self, window_function: WindowFunction) -> Self:
        return self.__class__(
            self._call,
            window_function,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            version=self._version,
        )

    def cum_sum(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("sum", reverse=reverse))

    def cum_max(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("max", reverse=reverse))

    def cum_min(self, *, reverse: bool) -> Self:
        return self._with_window_function(self._cum_window_func("min", reverse=reverse))

    def cum_count(self, *, reverse: bool) -> Self:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            descending = list(extend_bool(reverse, len(inputs.order_by)))
            nulls_first = list(extend_bool(not reverse, len(inputs.order_by)))
            return [
                self._window_expression(
                    expr.count(),
                    inputs.partition_by,
                    inputs.order_by,
                    descending=descending,
                    nulls_first=nulls_first,
                    rows_end=0,
                )
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def cum_prod(self, *, reverse: bool) -> Self:
        return self._with_window_function(
            self._cum_window_func("product", reverse=reverse)
        )

    def rolling_sum(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func("sum", window_size, min_samples, center=center)
        )

    def rolling_mean(self, window_size: int, *, min_samples: int, center: bool) -> Self:
        return self._with_window_function(
            self._rolling_window_func("mean", window_size, min_samples, center=center)
        )

    def rolling_var(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                "var", window_size, min_samples, ddof=ddof, center=center
            )
        )

    def rolling_std(
        self, window_size: int, *, min_samples: int, center: bool, ddof: int
    ) -> Self:
        return self._with_window_function(
            self._rolling_window_func(
                "std", window_size, min_samples, ddof=ddof, center=center
            )
        )

    def is_first_distinct(self) -> Self:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            descending = list(extend_bool(False, len(inputs.order_by)))
            return [
                self._window_expression(
                    F.row_number(),
                    (*inputs.partition_by, expr),
                    inputs.order_by,
                    descending=descending,
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def is_last_distinct(self) -> Self:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            descending = list(extend_bool(True, len(inputs.order_by)))
            nulls_first = list(extend_bool(False, len(inputs.order_by)))
            return [
                self._window_expression(
                    F.row_number(),
                    (*inputs.partition_by, expr),
                    inputs.order_by,
                    descending=descending,
                    nulls_first=nulls_first,
                )
                == lit(1)
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def is_unique(self) -> Self:
        def _is_unique(expr: Expression, *partition_by: str | Expression) -> Expression:
            return self._window_expression(
                expr.count(mode="all"), (expr, *partition_by)
            ) == lit(1)

        def _unpartitioned_is_unique(expr: Expression) -> Expression:
            return _is_unique(expr)

        def _partitioned_is_unique(
            df: DaftLazyFrame, inputs: WindowInputs
        ) -> Sequence[Expression]:
            assert not inputs.order_by  # noqa: S101
            return [_is_unique(expr, *inputs.partition_by) for expr in self(df)]

        return self._with_callable(_unpartitioned_is_unique)._with_window_function(
            _partitioned_is_unique
        )

    def diff(self) -> Self:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            window = self._window_expression
            descending = list(extend_bool(False, len(inputs.order_by)))
            nulls_first = list(extend_bool(True, len(inputs.order_by)))
            return [
                op.sub(
                    expr,
                    window(
                        F.lag(expr),
                        inputs.partition_by,
                        inputs.order_by,
                        descending=descending,
                        nulls_first=nulls_first,
                    ),
                )
                for expr in self(df)
            ]

        return self._with_window_function(func)

    def shift(self, n: int) -> Self:
        def func(df: DaftLazyFrame, inputs: WindowInputs) -> Sequence[Expression]:
            window = self._window_expression
            descending = list(extend_bool(False, len(inputs.order_by)))
            nulls_first = list(extend_bool(True, len(inputs.order_by)))
            return [
                window(
                    F.lag(expr, n),
                    inputs.partition_by,
                    inputs.order_by,
                    descending=descending,
                    nulls_first=nulls_first,
                )
                for expr in self(df)
            ]

        return self._with_window_function(func)

    @property
    def name(self) -> ExprNameNamespace:
        return ExprNameNamespace(self)

    @property
    def str(self) -> ExprStringNamespace:
        return ExprStringNamespace(self)

    drop_nulls = not_implemented()
    fill_nan = not_implemented()
    filter = not_implemented()
    ewm_mean = not_implemented()
    kurtosis = not_implemented()
    rank = not_implemented()
    map_batches = not_implemented()
    median = not_implemented()
    mode = not_implemented()
    quantile = not_implemented()
    replace_strict = not_implemented()
    unique = not_implemented()
    first = not_implemented()
    last = not_implemented()

    # namespaces
    dt = not_implemented()  # pyright: ignore[reportAssignmentType]
    cat = not_implemented()  # pyright: ignore[reportAssignmentType]
    list = not_implemented()  # pyright: ignore[reportAssignmentType]
    struct = not_implemented()  # pyright: ignore[reportAssignmentType]
