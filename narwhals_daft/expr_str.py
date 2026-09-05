from __future__ import annotations

import re
from typing import TYPE_CHECKING

import daft.functions as F
from daft import lit
from daft.expressions import col
from narwhals._utils import not_implemented
from narwhals.compliant import StringNamespace

if TYPE_CHECKING:
    from daft import Expression

    from narwhals_daft.expr import DaftExpr


_REGEX_METACHARACTERS = re.compile(r"[\\^$.|?*+()\[\]{}]")
"""Characters that make a pattern a regex rather than a literal, as in Polars."""


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
        if n < 0:
            return self.replace_all(value, pattern, literal=literal)
        if n == 0:
            return self.compliant
        is_literal_pattern = literal or not _REGEX_METACHARACTERS.search(pattern)
        if n > 1 and not (pattern and is_literal_pattern):
            # Re-scanning the remainder after a match changes the meaning of
            # anchors and word boundaries, and an empty pattern matches between
            # every character, so `n > 1` is limited to non-empty literal
            # patterns (like in Polars).
            kind = "regex" if pattern else "empty pattern"
            msg = f"{kind} replacement with 'n > 1' not yet supported"
            raise NotImplementedError(msg)
        # Daft only ships replace-all kernels. Anchoring the pattern behind a
        # lazy prefix group makes the first match addressable: group 0 spans
        # everything up to and including it, group 1 only the text before it,
        # and both are null when there is no match. `(?s)` lets the prefix span
        # newlines and the non-capturing wrapper keeps top-level alternation in
        # `pattern` contained.
        needle = re.escape(pattern) if literal else pattern
        anchored = f"(?s)^(.*?)(?:{needle})"

        def func(expr: Expression, value: Expression) -> Expression:
            # Splicing with `concat` inserts `value` verbatim, whereas the
            # replacement of `regexp_replace` expands `$1`/`\1` references and
            # drops backslashes.
            done, rest = lit(""), expr
            for _ in range(n):
                head = F.regexp_extract(rest, anchored, 0)
                prefix = F.regexp_extract(rest, anchored, 1)
                matched = head.not_null()
                done = F.when(
                    matched, F.concat(F.concat(done, prefix), value)
                ).otherwise(done)
                # `substr` yields null rather than "" once the offset reaches the end.
                rest = F.when(
                    matched, F.substr(rest, F.length(head)).fill_null("")
                ).otherwise(rest)
            return F.concat(done, rest)

        return self.compliant._with_elementwise(func, value=value)

    contains = not_implemented()
    to_datetime = not_implemented()
    zfill = not_implemented()
    pad_start = not_implemented()
    pad_end = not_implemented()
    to_time = not_implemented()
