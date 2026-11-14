"""
Day 4 - Customer Recommendation Engine
Squad Python Specialization - Data Science, AI & Analytics

Core Python only (csv, re, pathlib). No ML libraries.

How it works
  1. Each customer's purchase history is a set of product IDs.
  2. Two customers are compared with Jaccard similarity: |A & B| / |A | B|.
  3. The most similar customers (neighbours) vote for products the target
     customer has not bought. Each vote is weighted by the neighbour's similarity.
  4. Score = similarity_sum x (1 + popularity_share). See score() and README.md.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"

REQUIRED_COLUMNS = [
    "purchase_id", "customer_id", "customer_name", "product_id",
    "product_name", "category", "purchase_date", "quantity",
]
CSV_COLUMNS = [
    "customer_id", "customer_name", "rank", "product_id",
    "product_name", "category", "score", "supporting_customers",
]

_CUSTOMER_ID = re.compile(r"C\d+")
_PRODUCT_ID = re.compile(r"P\d+")
_WHOLE_NUMBER = re.compile(r"\d+")


class UnknownCustomerError(KeyError):
    """Raised when a customer ID is not in the engine."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else "unknown customer"


# ---------------------------------------------------------------------------
# Task 1: reading and cleaning
# ---------------------------------------------------------------------------
def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict], list[dict]]:
    """Return (valid_rows, rejected_rows).

    Cleaning rules (see README):
      Repaired  : surrounding whitespace; customer/product IDs upper-cased ('c106' -> 'C106').
      Rejected  : blank purchase_id, customer_id or product_id; malformed IDs;
                  quantity that is not a whole number greater than zero;
                  a purchase_id already seen (repeated row).

    Valid rows carry an int quantity and a 'repairs' list. Rejected rows are
    {'row': data row number, 'purchase_id': ..., 'reason': ..., 'raw': original row}.
    """
    try:
        f = open(path, newline="", encoding="utf-8-sig")
    except FileNotFoundError:
        raise FileNotFoundError(f"purchases file not found: {path}") from None

    valid: list[dict] = []
    rejected: list[dict] = []
    seen_ids: set[str] = set()

    with f:
        reader = csv.DictReader(f)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")

        for row_number, raw in enumerate(reader, start=1):
            clean = {k: (v or "").strip() for k, v in raw.items() if k is not None}
            repairs: list[str] = []
            reason = None

            for field, pattern in (("customer_id", _CUSTOMER_ID), ("product_id", _PRODUCT_ID)):
                original = (raw.get(field) or "").strip()
                upper = original.upper()
                if upper != original:
                    repairs.append(f"{field} {original!r} -> {upper!r}")
                clean[field] = upper

            if not clean["purchase_id"]:
                reason = "blank purchase_id"
            elif not clean["customer_id"]:
                reason = "blank customer_id"
            elif not clean["product_id"]:
                reason = "blank product_id"
            elif not _CUSTOMER_ID.fullmatch(clean["customer_id"]):
                reason = f"malformed customer_id {clean['customer_id']!r}"
            elif not _PRODUCT_ID.fullmatch(clean["product_id"]):
                reason = f"malformed product_id {clean['product_id']!r}"
            elif not _WHOLE_NUMBER.fullmatch(clean["quantity"]):
                reason = f"quantity {clean['quantity']!r} is not a whole number"
            elif int(clean["quantity"]) <= 0:
                reason = "quantity is zero"
            elif clean["purchase_id"] in seen_ids:
                reason = f"repeated row (purchase_id {clean['purchase_id']} already used)"

            if reason:
                rejected.append(
                    {"row": row_number, "purchase_id": clean["purchase_id"], "reason": reason, "raw": raw}
                )
                continue

            seen_ids.add(clean["purchase_id"])
            clean["quantity"] = int(clean["quantity"])
            clean["repairs"] = repairs
            valid.append(clean)
    return valid, rejected


# ---------------------------------------------------------------------------
# Task 3: similarity
# ---------------------------------------------------------------------------
def jaccard(a: set, b: set) -> float:
    """|A intersection B| / |A union B|.

    Two empty sets have no evidence of similarity, so the result is 0.0
    (this also avoids dividing by zero).
    """
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Tasks 2, 4-8: engine
# ---------------------------------------------------------------------------
class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers.

    All lookups are built once in __init__. No method re-scans the raw rows.
    """

    def __init__(
        self,
        purchases: list[dict],
        min_similarity: float = 0.0,
        *,
        k_neighbours: int = 3,
    ) -> None:
        if not purchases:
            raise ValueError("RecommendationEngine requires at least one purchase")
        if not 0.0 <= min_similarity <= 1.0:
            raise ValueError("min_similarity must be between 0 and 1")
        if k_neighbours < 1:
            raise ValueError("k_neighbours must be at least 1")

        self.min_similarity = min_similarity
        self.k_neighbours = k_neighbours

        histories: dict[str, set[str]] = {}
        names: dict[str, str] = {}
        products: dict[str, dict[str, str]] = {}

        for row in purchases:
            cid, pid = row["customer_id"], row["product_id"]
            histories.setdefault(cid, set()).add(pid)
            if not names.get(cid):
                names[cid] = row.get("customer_name") or cid
            info = products.setdefault(pid, {"name": "", "category": ""})
            if not info["name"]:
                info["name"] = row.get("product_name") or pid
            if not info["category"]:
                info["category"] = row.get("category") or "Unknown"

        # Popularity = number of DISTINCT customers who bought the product.
        popularity: dict[str, int] = {pid: 0 for pid in products}
        for items in histories.values():
            for pid in items:
                popularity[pid] += 1

        self._histories = histories
        self._names = names
        self._products = products
        self._popularity = popularity
        self._customer_ids = sorted(histories)
        self._n_customers = len(histories)

    # -- lookups -------------------------------------------------------------
    def _check(self, customer_id: str) -> None:
        if customer_id not in self._histories:
            raise UnknownCustomerError(f"unknown customer ID: {customer_id!r}")

    @property
    def customer_ids(self) -> list[str]:
        return list(self._customer_ids)

    @property
    def product_ids(self) -> list[str]:
        return sorted(self._products)

    def name(self, customer_id: str) -> str:
        self._check(customer_id)
        return self._names[customer_id]

    def product_info(self, product_id: str) -> dict[str, str]:
        return dict(self._products[product_id])

    def popularity(self, product_id: str) -> int:
        return self._popularity[product_id]

    def history(self, customer_id: str) -> set[str]:
        """A copy of the customer's purchased product IDs."""
        self._check(customer_id)
        return set(self._histories[customer_id])

    # -- similarity ----------------------------------------------------------
    def similarity(self, customer_a: str, customer_b: str) -> float:
        self._check(customer_a)
        self._check(customer_b)
        return jaccard(self._histories[customer_a], self._histories[customer_b])

    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """Return the k most similar other customers as (customer_id, score), best first.

        Excluded: the customer themself, anyone below min_similarity, and anyone with
        similarity 0 (sharing no product gives no evidence). Ties are broken by
        customer ID ascending.
        """
        self._check(customer_id)
        if k < 1:
            raise ValueError("k must be at least 1")
        mine = self._histories[customer_id]
        scored = [
            (other, jaccard(mine, items))
            for other, items in self._histories.items()
            if other != customer_id
        ]
        eligible = [(o, s) for o, s in scored if s > 0 and s >= self.min_similarity]
        eligible.sort(key=lambda pair: (-pair[1], pair[0]))
        return eligible[:k]

    # -- candidates ----------------------------------------------------------
    def _gather(self, customer_id: str, neighbours: list[tuple[str, float]]) -> dict[str, dict]:
        mine = self._histories[customer_id]
        candidates: dict[str, dict] = {}
        for neighbour, sim in neighbours:
            for pid in self._histories[neighbour] - mine:
                entry = candidates.setdefault(
                    pid,
                    {
                        "product_id": pid,
                        "supporters": [],
                        "support_count": 0,
                        "similarity_sum": 0.0,
                        "popularity": self._popularity[pid],
                        "fallback": False,
                    },
                )
                entry["supporters"].append((neighbour, sim))
                entry["support_count"] += 1
                entry["similarity_sum"] += sim
        return candidates

    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """Products bought by similar customers but not by this one, with the evidence for each.

        Evidence per product: supporters [(customer_id, similarity)], support_count,
        similarity_sum, popularity, fallback (False).

        Fallback, in two steps:
          1. If the top-k neighbours offer nothing new, widen to ALL eligible neighbours.
          2. If still nothing (new customer, no overlap with anyone), return every product
             the customer has not bought with fallback=True, ranked later by popularity alone.
        """
        self._check(customer_id)
        mine = self._histories[customer_id]

        candidates = self._gather(customer_id, self.similar_customers(customer_id, k))
        if not candidates:
            everyone = self.similar_customers(customer_id, max(1, self._n_customers))
            candidates = self._gather(customer_id, everyone)
        if candidates:
            return candidates

        return {
            pid: {
                "product_id": pid,
                "supporters": [],
                "support_count": 0,
                "similarity_sum": 0.0,
                "popularity": self._popularity[pid],
                "fallback": True,
            }
            for pid in self._products
            if pid not in mine
        }

    # -- ranking -------------------------------------------------------------
    def score(self, evidence: dict) -> float:
        """Ranking rule.

        popularity_share = customers who bought the product / all customers (0..1)

        Neighbour evidence : score = similarity_sum x (1 + popularity_share)
        Fallback evidence  : score = popularity_share

        similarity_sum adds up the Jaccard similarity of every supporting neighbour,
        so it rewards both how alike the supporters are and how many there are.
        The popularity multiplier (1.0 to 2.0) lifts widely bought products and acts
        as a tie-breaker between products with similar support.
        """
        share = evidence["popularity"] / self._n_customers
        if evidence["fallback"]:
            return share
        return evidence["similarity_sum"] * (1 + share)

    def recommend(self, customer_id: str, n: int = 3) -> list[dict]:
        """Return up to n ranked recommendations for one customer.

        Order: score descending, then popularity descending, then product ID ascending.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        candidates = self.candidate_products(customer_id, self.k_neighbours)
        ranked = sorted(
            candidates.values(),
            key=lambda e: (-round(self.score(e), 9), -e["popularity"], e["product_id"]),
        )[:n]

        rows = []
        for rank, e in enumerate(ranked, start=1):
            info = self._products[e["product_id"]]
            rows.append(
                {
                    "customer_id": customer_id,
                    "customer_name": self._names[customer_id],
                    "rank": rank,
                    "product_id": e["product_id"],
                    "product_name": info["name"],
                    "category": info["category"],
                    "score": round(self.score(e), 3),
                    "supporting_customers": [c for c, _ in e["supporters"]],
                    "support_count": e["support_count"],
                    "popularity": e["popularity"],
                    "fallback": e["fallback"],
                }
            )
        return rows

    def recommend_all(self, n: int = 3) -> list[dict]:
        """Recommendations for every customer. A customer with none gets one blank row."""
        rows: list[dict] = []
        for cid in self._customer_ids:
            recs = self.recommend(cid, n)
            if recs:
                rows.extend(recs)
            else:
                rows.append(
                    {
                        "customer_id": cid,
                        "customer_name": self._names[cid],
                        "rank": "",
                        "product_id": "",
                        "product_name": "",
                        "category": "",
                        "score": "",
                        "supporting_customers": [],
                        "support_count": 0,
                        "popularity": 0,
                        "fallback": False,
                    }
                )
        return rows

    # -- export --------------------------------------------------------------
    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                out = dict(row)
                out["score"] = f"{row['score']:.3f}" if row["score"] != "" else ""
                out["supporting_customers"] = ";".join(row["supporting_customers"])
                writer.writerow(out)
