"""Unit tests. Run from the project root with:  python -m unittest discover -s tests -v"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recommender import (  # noqa: E402
    CSV_COLUMNS,
    RecommendationEngine,
    UnknownCustomerError,
    jaccard,
    read_purchases,
)


def make_rows(history: dict[str, list[str]]) -> list[dict]:
    """Build purchase rows from {customer_id: [product_ids]}."""
    rows = []
    for cid, pids in history.items():
        for pid in pids:
            rows.append(
                {
                    "customer_id": cid,
                    "customer_name": f"Name-{cid}",
                    "product_id": pid,
                    "product_name": f"Product-{pid}",
                    "category": "Cat",
                }
            )
    return rows


class TestJaccard(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(jaccard({1, 2, 3}, {1, 2, 3}), 1.0)
        self.assertEqual(jaccard({1, 2}, {3, 4}), 0.0)
        self.assertAlmostEqual(jaccard({1, 2, 3}, {2, 3, 4}), 0.5)

    def test_empty_sets(self):
        self.assertEqual(jaccard(set(), set()), 0.0)
        self.assertEqual(jaccard({1}, set()), 0.0)

    def test_symmetric(self):
        self.assertEqual(jaccard({1, 2}, {2, 3, 4}), jaccard({2, 3, 4}, {1, 2}))


class TestReadPurchases(unittest.TestCase):
    HEADER = "purchase_id,customer_id,customer_name,product_id,product_name,category,purchase_date,quantity\n"

    def _read(self, body: str):
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, newline="", encoding="utf-8")
        tmp.write(self.HEADER + body)
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        return read_purchases(Path(tmp.name))

    def test_cleaning_rules(self):
        valid, rejected = self._read(
            "T1,C1,A,P1,X,Cat,2026-10-01,2\n"
            "T1,C1,A,P1,X,Cat,2026-10-01,2\n"      # repeated purchase_id
            "T2,C1,A,,,Cat,2026-10-01,1\n"          # blank product
            "T3,C1,A,P2,Y,Cat,2026-10-01,0\n"       # zero quantity
            "T4,c2,B,p3,Z,Cat,2026-10-01,1\n"       # lower-case IDs: repaired
            "T5,C3,C,P4,W,Cat,2026-10-01,two\n"     # non-numeric quantity
            "T6,CUST9,D,P5,V,Cat,2026-10-01,1\n"    # malformed customer id
        )
        self.assertEqual([r["purchase_id"] for r in valid], ["T1", "T4"])
        self.assertEqual(valid[1]["customer_id"], "C2")
        self.assertEqual(valid[1]["product_id"], "P3")
        self.assertEqual(len(valid[1]["repairs"]), 2)
        reasons = {r["purchase_id"]: r["reason"] for r in rejected}
        self.assertIn("repeated", reasons["T1"])
        self.assertEqual(reasons["T2"], "blank product_id")
        self.assertEqual(reasons["T3"], "quantity is zero")
        self.assertIn("not a whole number", reasons["T5"])
        self.assertIn("malformed customer_id", reasons["T6"])

    def test_missing_file_and_column(self):
        with self.assertRaises(FileNotFoundError):
            read_purchases(Path("nope.csv"))
        tmp = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        tmp.write("a,b\n1,2\n")
        tmp.close()
        self.addCleanup(Path(tmp.name).unlink)
        with self.assertRaises(ValueError):
            read_purchases(Path(tmp.name))


class TestEngine(unittest.TestCase):
    def setUp(self):
        self.history = {
            "C1": ["P1", "P2"],
            "C2": ["P1", "P2", "P3"],   # jaccard with C1 = 2/3
            "C3": ["P1", "P4"],         # jaccard with C1 = 1/3
            "C4": ["P5"],               # shares nothing with C1
            "C5": ["P1", "P4"],         # ties with C3
        }
        self.engine = RecommendationEngine(make_rows(self.history))

    def test_similarity_is_symmetric_with_unit_diagonal(self):
        self.assertAlmostEqual(self.engine.similarity("C1", "C2"), 2 / 3)
        self.assertEqual(self.engine.similarity("C1", "C2"), self.engine.similarity("C2", "C1"))
        for cid in self.engine.customer_ids:
            self.assertEqual(self.engine.similarity(cid, cid), 1.0)

    def test_similar_customers_order_ties_and_exclusions(self):
        result = self.engine.similar_customers("C1", k=5)
        ids = [c for c, _ in result]
        self.assertEqual(ids, ["C2", "C3", "C5"])        # C4 has similarity 0, C1 is self
        self.assertEqual(self.engine.similar_customers("C1", k=2)[1][0], "C3")  # tie -> lower ID

    def test_min_similarity_filters_neighbours(self):
        engine = RecommendationEngine(make_rows(self.history), min_similarity=0.5)
        self.assertEqual([c for c, _ in engine.similar_customers("C1", k=5)], ["C2"])

    def test_candidates_exclude_owned_products_and_list_supporters(self):
        candidates = self.engine.candidate_products("C1", k=3)
        self.assertEqual(set(candidates), {"P3", "P4"})
        self.assertEqual([c for c, _ in candidates["P4"]["supporters"]], ["C3", "C5"])
        self.assertEqual(candidates["P4"]["support_count"], 2)
        self.assertFalse(candidates["P4"]["fallback"])

    def test_score_combines_similarity_and_popularity(self):
        low = {"similarity_sum": 0.5, "popularity": 1, "fallback": False}
        high_sim = {"similarity_sum": 1.0, "popularity": 1, "fallback": False}
        popular = {"similarity_sum": 0.5, "popularity": 5, "fallback": False}
        self.assertGreater(self.engine.score(high_sim), self.engine.score(low))
        self.assertGreater(self.engine.score(popular), self.engine.score(low))
        self.assertAlmostEqual(self.engine.score(low), 0.5 * (1 + 1 / 5))

    def test_recommend_ranking_and_fields(self):
        recs = self.engine.recommend("C1", n=3)
        self.assertEqual([r["rank"] for r in recs], [1, 2])
        self.assertEqual(recs[0]["product_id"], "P4")  # 2 supporters beat 1 stronger supporter
        self.assertEqual(recs[0]["supporting_customers"], ["C3", "C5"])
        self.assertEqual(recs[1]["product_id"], "P3")

    def test_widens_to_all_neighbours_when_top_k_offer_nothing_new(self):
        engine = RecommendationEngine(
            make_rows(
                {
                    "C1": ["P1", "P2"],
                    "C2": ["P1", "P2"],           # identical: top neighbour, nothing new
                    "C3": ["P1", "P2"],           # identical: nothing new
                    "C4": ["P1", "P2"],           # identical: nothing new
                    "C5": ["P1", "P9"],           # lower similarity, offers P9
                }
            )
        )
        candidates = engine.candidate_products("C1", k=3)
        self.assertEqual(set(candidates), {"P9"})
        self.assertFalse(candidates["P9"]["fallback"])

    def test_no_neighbours_falls_back_to_popularity(self):
        recs = self.engine.recommend("C4", n=2)       # C4 shares nothing with anyone
        self.assertTrue(all(r["fallback"] for r in recs))
        self.assertEqual(recs[0]["product_id"], "P1")  # most popular product C4 lacks
        self.assertEqual(recs[0]["supporting_customers"], [])

    def test_customer_who_bought_everything_gets_nothing(self):
        engine = RecommendationEngine(make_rows({"C1": ["P1", "P2"], "C2": ["P1", "P2"]}))
        self.assertEqual(engine.recommend("C1"), [])

    def test_unknown_customer_raises_clear_error(self):
        for call in (
            lambda: self.engine.history("C99"),
            lambda: self.engine.similar_customers("C99"),
            lambda: self.engine.recommend("C99"),
            lambda: self.engine.similarity("C1", "C99"),
        ):
            with self.assertRaises(UnknownCustomerError) as ctx:
                call()
            self.assertIn("C99", str(ctx.exception))
        self.assertTrue(issubclass(UnknownCustomerError, KeyError))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            self.engine.recommend("C1", n=0)
        with self.assertRaises(ValueError):
            RecommendationEngine([])

    def test_engine_does_not_depend_on_raw_rows_after_construction(self):
        rows = make_rows(self.history)
        engine = RecommendationEngine(rows)
        before = engine.recommend("C1")
        rows.clear()
        self.assertEqual(engine.recommend("C1"), before)

    def test_history_returns_a_copy(self):
        h = self.engine.history("C1")
        h.add("P99")
        self.assertNotIn("P99", self.engine.history("C1"))

    def test_recommend_all_and_export_include_every_customer(self):
        engine = RecommendationEngine(make_rows({"C1": ["P1", "P2"], "C2": ["P1", "P2"], "C3": ["P1"]}))
        rows = engine.recommend_all(n=3)
        self.assertEqual({r["customer_id"] for r in rows}, {"C1", "C2", "C3"})
        blank = [r for r in rows if r["customer_id"] == "C1"]
        self.assertEqual(len(blank), 1)
        self.assertEqual(blank[0]["product_id"], "")  # nothing left to recommend

        tmp = Path(tempfile.mkdtemp()) / "out" / "recs.csv"
        engine.export_csv(rows, tmp)
        with open(tmp, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            written = list(reader)
        self.assertEqual({r["customer_id"] for r in written}, {"C1", "C2", "C3"})
        c3 = next(r for r in written if r["customer_id"] == "C3")
        self.assertRegex(c3["score"], r"^\d+\.\d{3}$")


if __name__ == "__main__":
    unittest.main()
