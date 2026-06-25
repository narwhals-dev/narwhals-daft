from __future__ import annotations

from collections.abc import Sequence
from itertools import chain
from typing import TYPE_CHECKING, Any, TypeVar

from narwhals.compliant import CompliantGroupBy

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from daft import Expression
    from typing_extensions import TypeIs

    from narwhals_daft.dataframe import DaftLazyFrame
    from narwhals_daft.expr import DaftExpr

_T = TypeVar("_T")


def is_sequence_but_not_str(sequence: Sequence[_T] | Any) -> TypeIs[Sequence[_T]]:
    return isinstance(sequence, Sequence) and not isinstance(sequence, str)


def is_sequence_of(obj: Any, tp: type[_T]) -> TypeIs[Sequence[_T]]:
    # Check if an object is a sequence of `tp`, only sniffing the first element.
    return bool(
        is_sequence_but_not_str(obj)
        and (first := next(iter(obj), None))
        and isinstance(first, tp)
    )


def _evaluate_aliases_single(frame: DaftLazyFrame, expr: DaftExpr) -> Sequence[str]:
    # NOTE: Ignore intermittent [False Negative]
    # Argument of type "CompliantFrameT@ImplExpr" cannot be assigned to parameter of type "CompliantFrameT@ImplExpr"
    #  Type "CompliantFrameT@ImplExpr" is not assignable to type "CompliantFrameT@ImplExpr"
    names = expr._evaluate_output_names(frame)  # pyright: ignore[reportArgumentType]
    return alias(names) if (alias := expr._alias_output_names) else names


def _evaluate_aliases_many(
    frame: DaftLazyFrame, exprs: Iterable[DaftExpr], /
) -> list[str]:
    it = (_evaluate_aliases_single(frame, expr) for expr in exprs)
    return list(chain.from_iterable(it))


class ParseKeysGroupBy(CompliantGroupBy["DaftLazyFrame", "DaftExpr"]):
    def _parse_keys(
        self, compliant_frame: DaftLazyFrame, keys: Sequence[DaftExpr] | Sequence[str]
    ) -> tuple[DaftLazyFrame, list[str], list[str]]:
        if is_sequence_of(keys, str):
            keys_str = list(keys)
            return compliant_frame, keys_str, keys_str.copy()
        return self._parse_expr_keys(compliant_frame, keys=keys)

    @staticmethod
    def _parse_expr_keys(
        compliant_frame: DaftLazyFrame, keys: Sequence[DaftExpr]
    ) -> tuple[DaftLazyFrame, list[str], list[str]]:
        """Parses key expressions to set up `.agg` operation with correct information.

        Since keys are expressions, it's possible to alias any such key to match
        other dataframe column names.

        In order to match polars behavior and not overwrite columns when evaluating keys:

        - We evaluate what the output key names should be, in order to remap temporary column
            names to the expected ones, and to exclude those from unnamed expressions in
            `.agg(...)` context (see https://github.com/narwhals-dev/narwhals/pull/2325#issuecomment-2800004520)
        - Create temporary names for evaluated key expressions that are guaranteed to have
            no overlap with any existing column name.
        - Add these temporary columns to the compliant dataframe.
        """
        tmp_name_length = max(len(str(c)) for c in compliant_frame.columns) + 1

        def _temporary_name(key: str) -> str:
            # 5 is the length of `__tmp`
            key_str = str(key)  # pandas allows non-string column names :sob:
            return f"_{key_str}_tmp{'_' * (tmp_name_length - len(key_str) - 5)}"

        keys_aliases = [
            _evaluate_aliases_single(compliant_frame, expr) for expr in keys
        ]
        safe_keys = [
            # multi-output expression cannot have duplicate names, hence it's safe to suffix
            key.name.map(_temporary_name)
            if (metadata := key._metadata) and metadata.expansion_kind.is_multi_output()
            # otherwise it's single named and we can use Expr.alias
            else key.alias(_temporary_name(new_names[0]))
            for key, new_names in zip(keys, keys_aliases, strict=True)
        ]
        return (
            compliant_frame.with_columns(*safe_keys),
            _evaluate_aliases_many(compliant_frame, safe_keys),
            list(chain.from_iterable(keys_aliases)),
        )


class DaftLazyGroupBy(ParseKeysGroupBy, CompliantGroupBy["DaftLazyFrame", "DaftExpr"]):
    def __init__(
        self,
        df: DaftLazyFrame,
        keys: Sequence[DaftExpr] | Sequence[str],
        /,
        *,
        drop_null_keys: bool,
    ) -> None:
        frame, self._keys, self._output_key_names = self._parse_keys(df, keys=keys)
        self._compliant_frame = (
            frame.drop_nulls(self._keys) if drop_null_keys else frame
        )

    def _evaluate_expr(self, expr: DaftExpr, /) -> Iterator[Expression]:
        output_names = expr._evaluate_output_names(self.compliant)
        aliases = (
            expr._alias_output_names(output_names)
            if expr._alias_output_names
            else output_names
        )
        native_exprs = expr(self.compliant)
        if expr._is_multi_output_unnamed():
            exclude = {*self._keys, *self._output_key_names}
            for native_expr, name, alias in zip(
                native_exprs, output_names, aliases, strict=True
            ):
                if name not in exclude:
                    yield expr._alias_native(native_expr, alias)
        else:
            for native_expr, alias in zip(native_exprs, aliases, strict=True):
                yield expr._alias_native(native_expr, alias)

    def _evaluate_exprs(self, exprs: Iterable[DaftExpr], /) -> Iterator[Expression]:
        for expr in exprs:
            yield from self._evaluate_expr(expr)

    def agg(self, *exprs: DaftExpr) -> DaftLazyFrame:
        result = (
            self.compliant.native.groupby(*self._keys).agg(*agg_columns)
            if (agg_columns := tuple(self._evaluate_exprs(exprs)))
            else self.compliant.native.select(*self._keys).drop_duplicates()
        )

        return self.compliant._with_native(result).rename(
            dict(zip(self._keys, self._output_key_names, strict=True))
        )
