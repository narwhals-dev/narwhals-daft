from __future__ import annotations

from typing import TYPE_CHECKING

import daft.functions as F
from narwhals._compliant.any_namespace import DateTimeNamespace
from narwhals._utils import not_implemented

if TYPE_CHECKING:
    from narwhals_daft.expr import DaftExpr


class ExprDateTimeNamesSpace(DateTimeNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr, /) -> None:
        self._compliant = expr

    @property
    def compliant(self) -> DaftExpr:
        return self._compliant

    def date(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.date(expr))

    def year(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.year(expr))

    def month(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.month(expr))

    def day(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.day(expr))

    def hour(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.hour(expr))

    def minute(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.minute(expr))

    def second(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.second(expr))

    def millisecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.millisecond(expr))

    def microsecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.microsecond(expr))

    def nanosecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.nanosecond(expr))

    def weekday(self) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.day_of_week(expr) + 1
        )  # daft is 0-6

    def ordinal_day(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.day_of_year(expr))

    to_string = not_implemented()
    replace_time_zone = not_implemented()
    convert_time_zone = not_implemented()
    timestamp = not_implemented()
    total_minutes = not_implemented()
    total_seconds = not_implemented()
    total_milliseconds = not_implemented()
    total_microseconds = not_implemented()
    total_nanoseconds = not_implemented()
    truncate = not_implemented()
    offset_by = not_implemented()
