from __future__ import annotations

import re
from typing import TYPE_CHECKING

import daft.functions as F
from daft import DataType, lit
from daft.expressions import col
from narwhals._utils import not_implemented
from narwhals.compliant import StringNamespace

if TYPE_CHECKING:
    from daft import Expression

    from narwhals_daft.expr import DaftExpr


_ISO_DATETIME_WITHOUT_SECONDS = r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}$"
# chrono format directives, grouped by the part of a timestamp they set.
_DATE_DIRECTIVE = re.compile(r"%[-_0^#]?[YCymbBhdeaAwuUWGgVjDxFvcs+]")
_TIME_DIRECTIVE = re.compile(r"%[-_0^#]?(?:\.?[369]?f|:{0,3}z|[HkIlPpMSRTXrcsZ+])")
_TIMEZONE_DIRECTIVE = re.compile(r"%[-_0^#]?(?::{0,3}z|[s+])")


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

    def starts_with(self, prefix: DaftExpr) -> DaftExpr:
        return self.compliant._with_elementwise(F.startswith, prefix=prefix)

    def ends_with(self, suffix: DaftExpr) -> DaftExpr:
        return self.compliant._with_elementwise(F.endswith, suffix=suffix)

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

    def _strip_chars_regex(self, pattern: str) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.regexp_replace(expr, pattern=pattern, replacement="")
        )

    def strip_chars_start(self, characters: str) -> DaftExpr:
        return self._strip_chars_regex(f"^{_char_class(characters)}+")

    def strip_chars_end(self, characters: str) -> DaftExpr:
        return self._strip_chars_regex(f"{_char_class(characters)}+$")

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

    def contains(self, pattern: DaftExpr, *, literal: bool) -> DaftExpr:
        if literal:
            return self.compliant._with_elementwise(F.contains, substr=pattern)
        return self.compliant._with_elementwise(F.regexp, pattern=pattern)

    def zfill(self, width: int) -> DaftExpr:
        if width <= 0:
            return self.compliant

        def func(expr: Expression) -> Expression:
            less_than_width = F.length(expr) < lit(width)
            sign = F.left(expr, 1)
            has_sign = (sign == lit("-")) | (sign == lit("+"))
            # `substr` yields null rather than "" once the offset reaches the end.
            digits = F.substr(expr, 1).fill_null("")
            return (
                F.when(
                    less_than_width & has_sign,
                    F.concat(sign, F.lpad(digits, width - 1, "0")),
                )
                .when(less_than_width, F.lpad(expr, width, "0"))
                .otherwise(expr)
            )

        return self.compliant._with_elementwise(func)

    def _pad(self, length: int, fill_char: str, *, start: bool) -> DaftExpr:
        if len(fill_char) != 1:
            msg = f"expected a string of length 1 as `fill_char`, got {fill_char!r}"
            raise ValueError(msg)
        if length <= 0:
            return self.compliant
        pad = F.lpad if start else F.rpad

        def func(expr: Expression) -> Expression:
            # `lpad`/`rpad` truncate strings longer than `length`.
            return F.when(
                F.length(expr) < lit(length), pad(expr, length, fill_char)
            ).otherwise(expr)

        return self.compliant._with_elementwise(func)

    def pad_start(self, length: int, fill_char: str) -> DaftExpr:
        return self._pad(length, fill_char, start=True)

    def pad_end(self, length: int, fill_char: str) -> DaftExpr:
        return self._pad(length, fill_char, start=False)

    replace = not_implemented()

    def to_datetime(self, format: str | None) -> DaftExpr:
        if format is not None:
            if not _TIME_DIRECTIVE.search(format):
                # `to_datetime` cannot build a timestamp from a date-only format.
                return self.compliant._with_elementwise(
                    lambda expr: F.to_date(expr, format).cast(DataType.timestamp("us"))
                )
            if _TIMEZONE_DIRECTIVE.search(format):
                # Parse offsets into UTC, like Polars. Without an explicit
                # timezone, an all-null column would also come back naive and
                # mismatch the planned `Timestamp[us; UTC]`.
                return self.compliant._with_elementwise(
                    lambda expr: F.to_datetime(expr, format, "UTC")
                )
            return self.compliant._with_elementwise(
                lambda expr: F.to_datetime(expr, format)
            )

        def func(expr: Expression) -> Expression:
            # Daft has no format inference, but casting parses the ISO 8601 /
            # RFC 3339 family; only the common form without seconds needs to be
            # normalized first.
            normalized = F.when(
                F.regexp(expr, _ISO_DATETIME_WITHOUT_SECONDS),
                F.concat(expr, lit(":00")),
            ).otherwise(expr)
            parsed = normalized.cast(DataType.timestamp("us"))
            # The cast silently yields null for anything it cannot parse.
            # Re-parsing only those rows with an explicit format makes them
            # raise at collect time instead.
            unparsed = F.when(expr.not_null() & parsed.is_null(), expr).otherwise(
                lit(None)
            )
            return F.coalesce(parsed, F.to_datetime(unparsed, "%Y-%m-%dT%H:%M:%S"))

        return self.compliant._with_elementwise(func)

    def to_time(self, format: str | None) -> DaftExpr:
        if format is None:

            def func(expr: Expression) -> Expression:
                parsed = expr.cast(DataType.time("us"))
                # As in `to_datetime`, make rows the cast could not parse raise.
                # A time-only format never yields a timestamp, so this only ever
                # raises for a non-null input.
                unparsed = F.when(expr.not_null() & parsed.is_null(), expr).otherwise(
                    lit(None)
                )
                return F.coalesce(parsed, F.time(F.to_datetime(unparsed, "%H:%M:%S")))

            return self.compliant._with_elementwise(func)
        if _DATE_DIRECTIVE.search(format):
            return self.compliant._with_elementwise(
                lambda expr: F.time(F.to_datetime(expr, format))
            )
        # `to_datetime` needs a full date to build a timestamp, so parse the
        # time behind a fixed one.
        return self.compliant._with_elementwise(
            lambda expr: F.time(
                F.to_datetime(F.concat(lit("1970-01-01 "), expr), f"%Y-%m-%d {format}")
            )
        )


def _char_class(characters: str) -> str:
    """Build a regex character class matching any of `characters` literally."""
    return "[" + "".join(f"\\x{{{ord(c):x}}}" for c in characters) + "]"
