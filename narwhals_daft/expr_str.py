from __future__ import annotations

from typing import TYPE_CHECKING

import daft.functions as F
from daft import lit
from daft.expressions import col
from narwhals._utils import not_implemented
from narwhals.compliant import StringNamespace

if TYPE_CHECKING:
    from daft import Expression

    from narwhals_daft.expr import DaftExpr


class ExprStringNamespace(StringNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr, /) -> None:
        self._compliant = expr

    @property
    def compliant(self) -> DaftExpr:
        return self._compliant

    def len_chars(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.length)

    def to_lowercase(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.lower)

    def to_titlecase(self) -> DaftExpr:
        def _to_titlecase(expr: Expression) -> Expression:
            if expr is None:
                return None
            lower_expr = F.lower(expr)
            extract_expr = F.regexp_extract_all(lower_expr, r"[a-z]*[^a-z]*", 0)
            capitalized_list = F.list_map(extract_expr, F.capitalize(col("")))
            return F.list_join(capitalized_list, delimiter="")

        return self.compliant._with_elementwise(_to_titlecase)

    def to_uppercase(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.upper)

    def to_date(self, format: str | None = None) -> DaftExpr:
        if format is None:
            format = "%Y-%m-%d"
        return self.compliant._with_elementwise(lambda expr: F.to_date(expr, format))

    def split(self, by: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.split(expr, by))

    def starts_with(self, prefix: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.startswith(expr, prefix))

    def ends_with(self, suffix: str) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.endswith(expr, suffix))

    def slice(self, offset: int, length: int | None = None) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            col_length = F.length(expr).cast(int)
            _offset = col_length + lit(offset) if offset < 0 else lit(offset)
            _length = lit(length) if length is not None else col_length
            return F.substr(expr, _offset, _length)

        return self.compliant._with_elementwise(func)

    def strip_chars(self, characters: str | None) -> DaftExpr:
        if characters is not None:
            # Feature request of `trim` in Daft
            # https://github.com/Eventual-Inc/Daft/issues/4021
            msg = "Non empty `characters` argument is not yet supported."
            raise NotImplementedError(msg)
        return self.compliant._with_elementwise(lambda expr: F.lstrip(F.rstrip(expr)))

    def replace_all(self, value: DaftExpr, pattern: str, *, literal: bool) -> DaftExpr:
        if literal:
            return self.compliant._with_elementwise(
                lambda expr, value: F.replace(expr, search=pattern, replacement=value),
                value=value,
            )
        return self.compliant._with_elementwise(
            lambda expr, value: F.regexp_replace(
                expr, pattern=pattern, replacement=value
            ),
            value=value,
        )

    def replace(
        self, value: DaftExpr, pattern: str, *, literal: bool, n: int
    ) -> DaftExpr:
        # `n` is the number of replacements: 1 = first occurrence, -1 (or
        # negative) = all occurrences. Daft only has replace-all kernels, so
        # implement first-`n` via find + substr reconstruction.
        if n == 0:
            return self.compliant
        if n < 0:
            return self.replace_all(value, pattern, literal=literal)
        # Check for multivalue + n>1 which narwhals does not support.
        # At compliant level `value` is always a DaftExpr (strs become lits),
        # so detect literal via metadata: if it's a literal, allow n>1 by
        # iterating; otherwise raise like other backends do.
        is_literal_value = bool(
            getattr(value, "_metadata", None) and value._metadata.is_literal
        )  # type: ignore[union-attr]
        if n > 1 and not is_literal_value:
            msg = "'n > 1' not yet supported for multivalue replacement."
            raise NotImplementedError(msg)

        def _replace_once(expr: Expression, value: Expression) -> Expression:
            expr_len = F.length(expr).cast("int64")
            if literal:
                idx = F.find(expr, pattern).cast("int64")
                match_len = lit(len(pattern)).cast("int64")
                suffix_start = idx + match_len
                prefix = F.when(idx == lit(0), lit("")).otherwise(
                    F.substr(expr, lit(0), idx)
                )
                suffix = F.when(suffix_start >= expr_len, lit("")).otherwise(
                    F.substr(expr, suffix_start, expr_len)
                )
                return F.when(idx == lit(-1), expr).otherwise(
                    F.when(expr.is_null() | idx.is_null(), expr).otherwise(
                        prefix + value + suffix
                    )
                )
            # Regex: extract first match, locate it, then splice.
            match = F.regexp_extract(expr, pattern)
            match_len = F.length(match).cast("int64")
            idx = F.find(expr, match).cast("int64")
            suffix_start = idx + match_len
            prefix = F.when(idx == lit(0), lit("")).otherwise(
                F.substr(expr, lit(0), idx)
            )
            suffix = F.when(suffix_start >= expr_len, lit("")).otherwise(
                F.substr(expr, suffix_start, expr_len)
            )
            return F.when(match.is_null(), expr).otherwise(prefix + value + suffix)

        result = self.compliant
        for _ in range(n):
            result = result._with_elementwise(
                _replace_once,
                value=value,  # type: ignore[arg-type]
            )
        return result

    contains = not_implemented()
    to_datetime = not_implemented()
    zfill = not_implemented()
    pad_start = not_implemented()
    pad_end = not_implemented()
    to_time = not_implemented()
