from __future__ import annotations

from typing import TYPE_CHECKING

import daft.functions as F
from daft import lit
from narwhals.compliant import ListNamespace

if TYPE_CHECKING:
    from daft import Expression
    from narwhals.typing import NonNestedliteral

    from narwhals_daft.expr import DaftExpr


class ExprListNamespace(ListNamespace["DaftExpr"]):
    def __init__(self, expr: DaftExpr, /) -> None:
        self._compliant = expr

    @property
    def compliant(self) -> DaftExpr:
        return self._compliant

    def len(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.list_count(expr, "all"))

    def min(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.list_min(expr))

    def max(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.list_max(expr))

    def mean(self) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.list_mean(expr))

    def sum(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            return F.when(F.list_count(expr, "valid") == lit(0), lit(0)).otherwise(
                F.list_sum(expr)
            )

        return self.compliant._with_elementwise(func)

    def sort(self, *, descending: bool, nulls_last: bool) -> DaftExpr:
        return self.compliant._with_elementwise(
            lambda expr: F.list_sort(expr, desc=descending, nulls_first=not nulls_last)
        )

    def unique(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            expr_distinct = F.list_distinct(expr)
            return F.when(
                F.list_count(expr, "null") == lit(0), expr_distinct
            ).otherwise(F.list_append(expr_distinct, lit(None)))

        return self.compliant._with_elementwise(func)

    def get(self, index: int) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.get(expr, key=index))

    def median(self) -> DaftExpr:
        def func(expr: Expression) -> Expression:
            sorted_expr = F.list_sort(expr, nulls_first=False)
            size = F.list_count(sorted_expr, mode="valid")
            mid_index = (size / lit(2)).cast("int").fill_null(0)
            odd_case = F.get(sorted_expr, key=mid_index)
            even_case = (
                F.get(sorted_expr, key=mid_index - lit(1))
                + F.get(sorted_expr, key=mid_index)
            ) / lit(2)
            return (
                F.when((size.is_null()) | (size == lit(0)), lit(None))
                .when(size % lit(2) == lit(1), odd_case)
                .otherwise(even_case)
            )

        return self.compliant._with_elementwise(func)

    def contains(self, item: NonNestedliteral) -> DaftExpr:
        return self.compliant._with_elementwise(lambda expr: F.is_in(expr, item))
