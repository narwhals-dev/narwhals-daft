from __future__ import annotations

from typing import TYPE_CHECKING

import daft.functions as F
from daft import DataType, lit
from narwhals._duration import Interval
from narwhals.compliant import DateTimeNamespace

if TYPE_CHECKING:
    from daft import Expression
    from narwhals._duration import IntervalUnit
    from narwhals.typing import TimeUnit

    from narwhals_daft.dataframe import DaftLazyFrame
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

    def replace_time_zone(self, time_zone: str | None) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.replace_time_zone(expr, time_zone)
        )

    def convert_time_zone(self, time_zone: str) -> DaftExpr:
        # `from_timezone` only applies to naive input, which Polars treats as UTC.
        return self.compliant._with_elementwise(
            lambda expr: F.convert_time_zone(expr, time_zone, from_timezone="UTC")
        )

    def timestamp(self, time_unit: TimeUnit) -> DaftExpr:
        def func(df: DaftLazyFrame, expr: Expression) -> Expression:
            dtype = df._native_dtype(expr)
            if not (dtype.is_date() or dtype.is_timestamp()):
                msg = "Input should be either of Date or Datetime type"
                raise TypeError(msg)
            return F.to_unix_epoch(expr, time_unit)

        return self.compliant._with_elementwise_frame(func)

    def truncate(self, every: str) -> DaftExpr:
        interval = Interval.parse(every)
        multiple, unit = interval.multiple, interval.unit

        def func(df: DaftLazyFrame, expr: Expression) -> Expression:
            dtype = df._native_dtype(expr)
            if unit in _MONTHS_PER_UNIT:
                # Count months since year 0, floor to the multiple, rebuild the date.
                every_months = multiple * _MONTHS_PER_UNIT[unit]
                index = F.year(expr).cast(_I64) * 12 + F.month(expr).cast(_I64) - 1
                index = index - index % every_months
                first_day = F.make_date(
                    index // 12, index % 12 + 1, _column_lit(expr, 1)
                )
                return _date_to(first_day, dtype)
            # Epoch-aligned, in local time for time zone aware input, like Polars.
            native_unit = f"{multiple} {_DATE_TRUNC_UNITS[unit]}"
            if dtype.is_date():
                return F.date_trunc(native_unit, expr.cast(_TS_US)).cast(
                    DataType.date()
                )
            step = multiple * _NS_PER_UNIT[unit]
            column_step = _NS_PER_UNIT[str(dtype.timeunit)]
            if step < column_step:
                # Daft panics on steps finer than the column's unit. When the step
                # divides it, every value is already on the grid.
                if column_step % step == 0:
                    return expr
                expr = expr.cast(DataType.timestamp("ns", dtype.timezone))
            return F.date_trunc(native_unit, expr)

        return self.compliant._with_elementwise_frame(func)

    def offset_by(self, by: str) -> DaftExpr:
        interval = Interval.parse_no_constraints(by)
        multiple, unit = interval.multiple, interval.unit

        def func(df: DaftLazyFrame, expr: Expression) -> Expression:
            dtype = df._native_dtype(expr)
            time_unit = "us" if dtype.is_date() else str(dtype.timeunit)
            if unit not in {"d", *_MONTHS_PER_UNIT}:
                return _add_duration(expr, multiple, unit, time_unit)
            # Calendar units shift the local wall time, so DST changes don't
            # move the time of day.
            time_zone = None if dtype.is_date() else dtype.timezone
            local = F.replace_time_zone(expr, None) if time_zone else expr
            if unit == "d":
                shifted = _add_duration(local, multiple, unit, time_unit)
            else:
                months = _column_lit(local, multiple * _MONTHS_PER_UNIT[unit])
                # `add_months` clamps to the end of the month and returns a Date.
                shifted = F.add_months(local, months)
                if not dtype.is_date():
                    time_of_day = local - F.date_trunc("1 day", local)
                    shifted = shifted.cast(DataType.timestamp(time_unit)) + time_of_day
            return F.replace_time_zone(shifted, time_zone) if time_zone else shifted

        return self.compliant._with_elementwise_frame(func)


_I64 = DataType.int64()
_TS_US = DataType.timestamp("us")
_MONTHS_PER_UNIT: dict[IntervalUnit, int] = {"mo": 1, "q": 3, "y": 12}
_DATE_TRUNC_UNITS: dict[IntervalUnit, str] = {
    "ns": "nanoseconds",
    "us": "microseconds",
    "ms": "milliseconds",
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}
_NS_PER_UNIT: dict[str, int] = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60 * 1_000_000_000,
    "h": 3_600 * 1_000_000_000,
    "d": 86_400 * 1_000_000_000,
}


def _column_lit(expr: Expression, value: int) -> Expression:
    """An Int64 constant shaped like `expr`.

    With a literal argument, `make_date` and `add_months` return the first row's
    result for every row (Daft 0.7.24), a column-shaped one avoids that.
    """
    return expr.is_null().cast(_I64) * 0 + lit(value)


def _date_to(date: Expression, dtype: DataType) -> Expression:
    """Cast a Date to `dtype`: midnight in local time for a time zone aware one."""
    if dtype.is_date():
        return date
    result = date.cast(DataType.timestamp(str(dtype.timeunit)))
    return F.replace_time_zone(result, dtype.timezone) if dtype.timezone else result


def _add_duration(
    expr: Expression, multiple: int, unit: IntervalUnit, time_unit: str
) -> Expression:
    """Add `multiple` `unit`s, expressed in the column's own time unit."""
    nanoseconds = multiple * _NS_PER_UNIT[unit]
    if nanoseconds % _NS_PER_UNIT[time_unit]:
        # Not representable in the column's unit, so go to nanoseconds.
        expr, time_unit = expr.cast(DataType.timestamp("ns")), "ns"
    count = nanoseconds // _NS_PER_UNIT[time_unit]
    return expr + lit(count).cast(DataType.duration(time_unit))
