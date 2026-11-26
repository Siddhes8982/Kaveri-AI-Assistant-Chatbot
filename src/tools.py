"""Read-only SQL tools backed by the configured Kaveri sales database."""
from __future__ import annotations

import os
from decimal import Decimal
from functools import lru_cache

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from dotenv import load_dotenv


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Return the shared SQLAlchemy engine using the existing DATABASE_URL."""
    load_dotenv()
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError("DATABASE_URL is not configured")
    return create_engine(url, future=True)


def get_revenue(country: str, year: int) -> str:
    """Return actual total order revenue for a customer country and calendar year."""
    if not isinstance(country, str) or not country.strip():
        raise ValueError("country must be a non-empty string")
    if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= 9999:
        raise ValueError("year must be a valid four-digit integer")
    query = text("""
        SELECT COUNT(o.order_id) AS order_count, SUM(o.total_amount) AS revenue
        FROM orders AS o
        JOIN customers AS c ON c.customer_id = o.customer_id
        WHERE TRIM(c.country) = :country
          AND o.order_date >= :start_date
          AND o.order_date < :end_date
    """)
    params = {"country": country.strip(), "start_date": f"{year:04d}-01-01",
              "end_date": f"{year + 1:04d}-01-01"}
    try:
        with get_engine().connect() as connection:
            row = connection.execute(query, params).one()
    except Exception as error:
        raise RuntimeError("Revenue database query failed") from error
    if not row.order_count:
        return f"No revenue records found for {country.strip()} in {year}."
    amount = Decimal(str(row.revenue or 0)).quantize(Decimal("0.01"))
    return f"USD {amount}"
