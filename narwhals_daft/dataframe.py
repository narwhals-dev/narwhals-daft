from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import daft
import daft.exceptions
import daft.functions as F
from daft import Expression
from narwhals._sql.dataframe import SQLLazyFrame
from narwhals._utils import (
    Implementation,
    ValidateBackendVersion,
    Version,
    check_column_names_are_unique,
    extend_bool,
    generate_temporary_column_name,
    not_implemented,
    parse_columns_to_drop,
)
from narwhals.exceptions import (
    ColumnNotFoundError,
    DuplicateError,
    MultiOutputExpressionError,
)

from narwhals_daft.group_by import DaftLazyGroupBy
from narwhals_daft.utils import evaluate_exprs, lit, native_to_narwhals_dtype

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence
    from io import BytesIO
    from pathlib import Path
    from types import ModuleType

    from narwhals._utils import _LimitedContext
    from narwhals.compliant import CompliantDataFrame
    from narwhals.dataframe import LazyFrame
    from narwhals.dtypes import DType
    from narwhals.typing import JoinStrategy
    from typing_extensions import TypeIs

    from narwhals_daft.expr import DaftExpr, WindowInputs
    from narwhals_daft.namespace import DaftNamespace


class DaftLazyFrame(
    SQLLazyFrame["DaftExpr", "daft.DataFrame", "LazyFrame[daft.DataFrame]"],
    ValidateBackendVersion,
):
    _implementation = Implementation.UNKNOWN

    def __init__(
        self,
        native_dataframe: daft.DataFrame,
        *,
        version: Version,
        validate_backend_version: bool = False,
    ) -> None:
        self._native_frame: daft.DataFrame = native_dataframe
        self._version = version
        self._cached_schema: dict[str, DType] | None = None
        self._cached_columns: list[str] | None = None
        if validate_backend_version:  # pragma: no cover
            self._validate_backend_version()

    @staticmethod
    def _is_native(obj: daft.DataFrame | Any) -> TypeIs[daft.DataFrame]:
        return isinstance(obj, daft.DataFrame)

    @classmethod
    def from_native(
        cls, data: daft.DataFrame, /, *, context: _LimitedContext
    ) -> DaftLazyFrame:
        return cls(data, version=context._version)

    def to_narwhals(self) -> LazyFrame[daft.DataFrame]:
        return self._version.lazyframe(self, level="lazy")

    def __native_namespace__(self) -> ModuleType:
        import daft

        return daft

    def __narwhals_namespace__(self) -> DaftNamespace:
        from narwhals_daft.namespace import DaftNamespace

        return DaftNamespace(version=self._version)

    def __narwhals_lazyframe__(self) -> DaftLazyFrame:
        return self

    def _with_version(self, version: Version) -> DaftLazyFrame:
        return self.__class__(self._native_frame, version=version)

    def _with_native(self, df: daft.DataFrame) -> DaftLazyFrame:
        return self.__class__(df, version=self._version)

    def _iter_columns(self) -> Iterator[daft.Expression]:
        return iter(self._native_frame.columns)

    def _evaluate_expr(self, expr: DaftExpr) -> Expression:
        result: Sequence[Expression] = expr(self)
        assert len(result) == 1  # debug assertion  # noqa: S101
        return result[0]

    def _evaluate_window_expr(
        self, expr: DaftExpr, /, window_inputs: WindowInputs
    ) -> Expression:
        result = expr.window_function(self, window_inputs)
        if len(result) != 1:  # pragma: no cover
            msg = "multi-output expressions not allowed in this context"
            raise MultiOutputExpressionError(msg)
        return result[0]

    @property
    def columns(self) -> list[str]:
        if self._cached_columns is None:
            self._cached_columns = (
                list(self.schema)
                if self._cached_schema is not None
                else self.native.column_names
            )
        return self._cached_columns

    def collect(
        self, backend: ModuleType | Implementation | str | None, **kwargs: Any
    ) -> CompliantDataFrame[Any, Any, Any, Any]:
        if backend is None or backend is Implementation.PYARROW:
            from narwhals._arrow.dataframe import ArrowDataFrame

            return ArrowDataFrame(
                native_dataframe=self._native_frame.to_arrow(),
                validate_backend_version=True,
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.PANDAS:
            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                native_dataframe=self._native_frame.to_pandas(),
                implementation=Implementation.PANDAS,
                validate_backend_version=True,
                version=self._version,
                validate_column_names=True,
            )

        if backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import
            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                df=cast("pl.DataFrame", pl.from_arrow(self._native_frame.to_arrow())),
                validate_backend_version=True,
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def simple_select(self, *column_names: str) -> DaftLazyFrame:
        return self._with_native(self._native_frame.select(*column_names))

    def aggregate(self, *exprs: DaftExpr) -> DaftLazyFrame:
        new_columns_map = evaluate_exprs(self, *exprs)
        return self._with_native(
            self._native_frame.agg([val.alias(col) for col, val in new_columns_map])
        )

    def select(self, *exprs: DaftExpr) -> DaftLazyFrame:
        new_columns_map = evaluate_exprs(self, *exprs)
        if not new_columns_map:
            msg = "At least one expression must be passed to LazyFrame.select"
            raise ValueError(msg)
        try:
            return self._with_native(
                self._native_frame.select(
                    *(val.alias(col) for col, val in new_columns_map)
                )
            )
        except daft.exceptions.DaftCoreException as e:
            if "duplicate" in str(e):  # pragma: no cover
                raise DuplicateError(e) from None
            if "not found" in str(e):
                msg = (
                    f"{e!s}\n\nHint: Did you mean one of these columns: {self.columns}?"
                )
                raise ColumnNotFoundError(msg) from e
            raise

    def with_columns(self, *exprs: DaftExpr) -> DaftLazyFrame:
        new_columns_map = dict(evaluate_exprs(self, *exprs))
        return self._with_native(self._native_frame.with_columns(new_columns_map))

    def _filter(self, predicate: DaftExpr) -> DaftLazyFrame:
        # `[0]` is safe as the predicate's expression only returns a single column
        mask = predicate._call(self)[0]
        return self._with_native(self._native_frame.filter(mask))

    @property
    def schema(self) -> dict[str, DType]:
        if self._cached_schema is None:
            # Note: prefer `self._cached_schema` over `functools.cached_property`
            # due to Python3.13 failures.
            self._cached_schema = {
                field.name: native_to_narwhals_dtype(field.dtype, self._version)
                for field in (self._native_frame.schema())
            }
        return self._cached_schema

    def collect_schema(self) -> dict[str, DType]:
        return {
            field.name: native_to_narwhals_dtype(field.dtype, self._version)
            for field in self._native_frame.schema()
        }

    def drop(self, columns: Sequence[str], *, strict: bool) -> DaftLazyFrame:
        columns_to_drop = parse_columns_to_drop(self, columns, strict=strict)
        selection = [col for col in self.columns if col not in columns_to_drop]
        return self._with_native(self._native_frame.select(*selection))

    def head(self, n: int) -> DaftLazyFrame:
        return self._with_native(self._native_frame.limit(n))

    def sort(
        self, *by: str, descending: bool | Sequence[bool], nulls_last: bool
    ) -> DaftLazyFrame:
        return self._with_native(
            self._native_frame.sort(
                list(by),
                desc=descending if isinstance(descending, bool) else list(descending),
                nulls_first=not nulls_last,
            )
        )

    def top_k(
        self, k: int, *, by: Iterable[str], reverse: bool | Sequence[bool]
    ) -> DaftLazyFrame:
        descending = [not x for x in extend_bool(reverse, len(list(by)))]
        return self._with_native(
            self._native_frame.sort(list(by), desc=descending, nulls_first=False).limit(
                k
            )
        )

    def drop_nulls(self, subset: Sequence[str] | None) -> DaftLazyFrame:
        if subset:
            return self._with_native(self._native_frame.drop_null(*subset))
        return self._with_native(self._native_frame.drop_null())

    def rename(self, mapping: Mapping[str, str]) -> DaftLazyFrame:
        selection = [
            daft.col(col).alias(mapping[col]) if col in mapping else col
            for col in self.columns
        ]
        return self._with_native(self.native.select(*selection))

    def unique(
        self, subset: Sequence[str] | None, *, keep: str, order_by: Sequence[str] | None
    ) -> DaftLazyFrame:
        subset_ = subset or self.columns
        tmp_name = generate_temporary_column_name(8, self.columns, prefix="row_index_")
        window = daft.Window().partition_by(*subset_)
        if order_by and keep == "last":
            window = window.order_by(*order_by, desc=True, nulls_first=False)
        elif order_by:
            window = window.order_by(*order_by, desc=False, nulls_first=True)
        if keep == "none":
            expr = daft.col(self.columns[0]).count(mode="all").over(window)
        else:
            expr = F.row_number().over(window)
        df = self.native.with_column(tmp_name, expr).filter(
            daft.col(tmp_name) == lit(1)
        )
        df = df.select(*(x for x in df.column_names if x != tmp_name))
        return self._with_native(df)

    def join(
        self,
        other: DaftLazyFrame,
        how: JoinStrategy,
        left_on: Sequence[str] | None,
        right_on: Sequence[str] | None,
        suffix: str,
    ) -> DaftLazyFrame:
        if how == "cross":
            return self._with_native(
                self.native.join(other.native, how="cross", prefix="", suffix=suffix)
            )
        left_columns = self.columns
        right_columns = other.columns

        right_on_ = list(right_on) if right_on is not None else []
        left_on_ = list(left_on) if left_on is not None else []

        # create a mapping for columns on other
        # `right_on` columns will be renamed as `left_on`
        # the remaining columns will be either added the suffix or left unchanged.
        right_cols_to_rename = (
            [c for c in right_columns if c not in right_on_]
            if how != "full"
            else right_columns
        )

        rename_mapping = {
            **dict(zip(right_on_, left_on_, strict=True)),
            **{
                colname: f"{colname}{suffix}" if colname in left_columns else colname
                for colname in right_cols_to_rename
            },
        }
        plx = self.__narwhals_namespace__()
        other_native = other.select(
            *[plx.col(old).alias(new) for old, new in rename_mapping.items()]
        ).native
        col_order = left_columns.copy()

        if how in {"inner", "left", "cross"}:
            col_order.extend(
                rename_mapping[colname]
                for colname in right_columns
                if colname not in right_on_
            )
        elif how == "full":
            col_order.extend(rename_mapping.values())

        check_column_names_are_unique(col_order)

        right_on_remapped = [rename_mapping[c] for c in right_on_]
        how_native: Literal["inner", "left", "semi", "anti", "outer"] = (
            "outer" if how == "full" else how
        )

        return self._with_native(
            self.native.join(
                other_native,
                left_on=[daft.col(x) for x in left_on_],
                right_on=[daft.col(x) for x in right_on_remapped],
                how=how_native,
            ).select(*col_order)
        )

    def unpivot(
        self,
        on: Sequence[str] | None,
        index: Sequence[str] | None,
        variable_name: str,
        value_name: str,
    ) -> DaftLazyFrame:
        index_ = [] if index is None else index
        on_ = [c for c in self.columns if c not in index_] if on is None else on
        return self._with_native(
            self._native_frame.unpivot(
                ids=index_,
                values=on_,
                variable_name=variable_name,
                value_name=value_name,
            )
        )

    def with_row_index(self, name: str, order_by: Sequence[str]) -> DaftLazyFrame:
        row_index_expr = (
            (
                daft.functions.row_number().over(
                    daft.Window().partition_by(lit(1)).order_by(*order_by)
                )
                - 1
            )
            if order_by
            else daft.functions.monotonically_increasing_id()
        )
        return self._with_native(
            self.native.select(row_index_expr.alias(name), *self.columns)
        )

    def group_by(
        self, keys: Sequence[str] | Sequence[DaftExpr], *, drop_null_keys: bool
    ) -> DaftLazyGroupBy:
        return DaftLazyGroupBy(self, keys, drop_null_keys=drop_null_keys)

    def explode(self, columns: Sequence[str]) -> DaftLazyFrame:
        return self._with_native(self.native.explode(*columns))

    def sink_parquet(self, file: str | Path | BytesIO) -> None:
        self.native.write_parquet(str(file))

    gather_every = not_implemented.deprecated(
        "`LazyFrame.gather_every` is deprecated and will be removed in a future version."
    )
    join_asof = not_implemented()
    tail = not_implemented.deprecated(
        "`LazyFrame.tail` is deprecated and will be removed in a future version."
    )
