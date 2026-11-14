"""Run the recommendation engine end to end."""
import sys
from pathlib import Path

from recommender import DATA_PATH, OUTPUT_PATH, RecommendationEngine, read_purchases

DEMO_CUSTOMERS = ["C101", "C113"]
N_RECOMMENDATIONS = 3


def print_matrix(engine: RecommendationEngine, size: int = 5) -> None:
    ids = engine.customer_ids[:size]
    print(f"Similarity matrix (first {len(ids)} customers)")
    print("      " + "".join(f"{c:>7}" for c in ids))
    for a in ids:
        print(f"{a:<6}" + "".join(f"{engine.similarity(a, b):>7.2f}" for b in ids))
    print()


def print_customer(engine: RecommendationEngine, customer_id: str) -> None:
    bought = sorted(engine.history(customer_id))
    print(f"Customer {customer_id} ({engine.name(customer_id)}) bought: {{{', '.join(bought)}}}")
    print("Most similar customers:")
    neighbours = engine.similar_customers(customer_id, engine.k_neighbours)
    if not neighbours:
        print("  none (no other customer shares a product)")
    for other, sim in neighbours:
        print(f"  {other}  jaccard={sim:.2f}")
    print("Recommendations:")
    recs = engine.recommend(customer_id, N_RECOMMENDATIONS)
    if not recs:
        print("  none (nothing left to recommend)")
    for r in recs:
        if r["fallback"]:
            why = f"it is popular ({r['popularity']} customers bought it)"
        else:
            n = r["support_count"]
            why = f"{n} similar customer{'s' if n != 1 else ''} bought it"
        print(f"  {r['rank']}. {r['product_name']} ({r['category']})  score={r['score']:.3f}  because {why}")
    print()


def main(path: Path = DATA_PATH, demo_ids: list[str] | None = None) -> int:
    try:
        valid, rejected = read_purchases(path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Rejected rows ({len(rejected)}):")
    for r in rejected:
        print(f"  row {r['row']} ({r['purchase_id'] or 'no id'}): {r['reason']}")
    repairs = [(row["purchase_id"], fix) for row in valid for fix in row["repairs"]]
    print(f"Repaired values ({len(repairs)}):")
    for pid, fix in repairs:
        print(f"  {pid}: {fix}")
    print()

    try:
        engine = RecommendationEngine(valid)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    print_matrix(engine)
    for cid in demo_ids or DEMO_CUSTOMERS:
        try:
            print_customer(engine, cid)
        except KeyError as exc:
            print(f"ERROR: {exc}\n")

    rows = engine.recommend_all(n=N_RECOMMENDATIONS)
    engine.export_csv(rows, OUTPUT_PATH)

    written = sum(1 for r in rows if r["product_id"])
    blank = sum(1 for r in rows if not r["product_id"])
    print(f"Purchases read: {len(valid) + len(rejected)} | used: {len(valid)} | rejected: {len(rejected)}")
    print(
        f"Customers: {len(engine.customer_ids)} | Products: {len(engine.product_ids)} | "
        f"Recommendations written: {written}"
    )
    if blank:
        print(f"Customers with no possible recommendation (blank row written): {blank}")
    print(f"Saved to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main(demo_ids=sys.argv[1:] or None))
