from __future__ import annotations

from typing import TYPE_CHECKING

import daft.functions as F
from narwhals._utils import not_implemented
from narwhals.compliant import DateTimeNamespace

if TYPE_CHECKING:
    from narwhals_daft.expr import DaftExpr


class ExprDateTimeNamesSpace(DateTimeNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr, /) -> None:
        self._compliant = expr

    @property
    def compliant(self) -> DaftExpr:
        return self._compliant

    def date(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.date)

    def year(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.year)

    def month(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.month)

    def day(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.day)

    def hour(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.hour)

    def minute(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.minute)

    def second(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.second)

    def millisecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.millisecond)

    def microsecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.microsecond)

    def nanosecond(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.nanosecond)

    def weekday(self) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.day_of_week(expr) + 1
        )  # daft is 0-6

    def ordinal_day(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.day_of_year)

    def to_string(self, format: str | None) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.strftime(expr, format))

    def total_minutes(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.total_minutes)

    def total_seconds(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.total_seconds)

    def total_milliseconds(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.total_milliseconds)

    def total_microseconds(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.total_microseconds)

    def total_nanoseconds(self) -> DaftExpr:
        return self.compliant._with_elementwise(F.total_nanoseconds)

    replace_time_zone = not_implemented()
    convert_time_zone = not_implemented()
    timestamp = not_implemented()
    truncate = not_implemented()
    offset_by = not_implemented()
