# Day 4: Customer Recommendation Engine

"Customers like you also bought" for Mama Mboga Online. Built from first principles: purchase histories as sets, Jaccard similarity, and an explainable ranking rule. Standard library only. No ML libraries.

## How to run

```bash
python main.py                  # full run: cleaning report, matrix, two demo customers, CSV export
python main.py C105 C110        # demo any customers instead
python -m unittest discover -s tests -v
```

Requires Python 3.9+. No packages to install. Output goes to `output/recommendations.csv`.

## Approach

| Task | Implementation |
|---|---|
| 1. Read | `read_purchases` uses `csv.DictReader`, returns `(valid_rows, rejected_rows)`. Each rejection has the data row number, purchase ID and a reason. Repairs are recorded on each valid row. |
| 2. Structures | `__init__` builds everything once: customer to set of products, customer names, product name and category, and product popularity. No method re-scans raw rows. A test clears the input list after construction and results are unchanged. |
| 3. Jaccard | `jaccard(a, b)` is `len(a & b) / len(a | b)`. `similarity()` wraps it. `main.py` prints the matrix for the first five customers. |
| 4. Neighbours | `similar_customers` returns `(customer_id, score)`, best first. Excludes the customer, anyone below `min_similarity`, and anyone with similarity 0. Ties break by customer ID ascending. |
| 5. Candidates | `candidate_products` collects products neighbours bought and the customer did not, with the supporting neighbours for each. |
| 6. Ranking | `score` and `recommend`. See the scoring rule below. |
| 7. Class | `RecommendationEngine(purchases).recommend_all(n=3)`. Unknown customers raise `UnknownCustomerError`, a `KeyError` subclass with a clear message. |
| 8. Export | `export_csv` writes the expected columns, score to 3 dp, supporters joined with `;`. |

## Data-cleaning rules

The file has 63 rows. 60 are used. 3 are rejected. 1 value is repaired.

| Case | Decision | Justification |
|---|---|---|
| Repeated row (T9010 twice) | Keep the first, reject the second. Rule: a `purchase_id` already seen. | A repeated purchase ID is one purchase recorded twice. Counting it would only inflate quantity, and sets ignore quantity anyway. |
| Blank product (T9020, row 21) | Rejected: blank product_id. | Without a product there is nothing to recommend or compare. |
| Zero quantity (T9033, row 34) | Rejected: quantity is zero. | Nothing was bought. Keeping it would put an unbought product in the customer's history and exclude it from their recommendations. |
| Inconsistent ID (`c106`, T9041) | Repaired to `C106`. | The customer and name match C106's other row. IDs and product IDs are trimmed and upper-cased. |
| Quantity not a whole number, malformed IDs | Rejected. | Not present in this file. Covered by the rules and tests. |

Quantity is used only for validity. Purchase date is not used.

## Ranking rule

For each product the customer has not bought:

```
popularity_share = customers who bought the product / all customers
score = similarity_sum x (1 + popularity_share)
```

`similarity_sum` adds the Jaccard similarity of every supporting neighbour. It rewards both how alike the supporters are and how many there are. The popularity multiplier (1.0 to 2.0) lifts widely bought products and separates products with similar support. Ties break by popularity descending, then product ID ascending.

**Worked example: customer C101 (Wanjiru), who bought P04 Blue Band and P08 Ketepa Tea.**

Neighbours (top 3): C117 (0.50), C109 (0.33), C113 (0.17).

| Product | Supporters | similarity_sum | Popularity | Score |
|---|---|---|---|---|
| P03 Festive Bread | C117, C109, C113 | 0.50 + 0.33 + 0.17 = 1.00 | 7 of 20 = 0.35 | 1.00 x 1.35 = **1.350** |
| P10 Pampers Size 3 | C117 | 0.50 | 6 of 20 = 0.30 | 0.50 x 1.30 = **0.650** |
| P01 Kienyeji Eggs | C113 | 0.17 | 5 of 20 = 0.25 | 0.167 x 1.25 = **0.208** |

Festive Bread ranks first because all three neighbours bought it. Pampers beats Eggs because its supporter is more similar and it is more popular.

## Edge cases

- **New customer or no overlap with anyone:** fall back to the most popular products the customer has not bought. Score is `popularity_share` and the supporters column is blank. Fallback scores are not comparable with neighbour scores, but the two are never mixed for one customer.
- **Top 3 neighbours offer nothing new:** the engine first widens to all eligible neighbours before using the popularity fallback. This applied to C107, C108 and C120. No customer needed the popularity fallback on this data.
- **Customer who bought everything:** no recommendation exists. One row with blank product columns is written, so nobody disappears from the file.
- **Two empty sets:** `jaccard` returns 0.0, which avoids division by zero. No customer has an empty history, since customers exist only through valid purchases.
- **Unknown customer ID:** `UnknownCustomerError` (a `KeyError`).

## Findings

1. All 20 customers appear in `recommendations.csv`: 52 recommendation rows, 0 blank rows.
2. Customers get fewer than 3 recommendations when their neighbours have little to add. C104 gets 1 and C106, C110, C111, C114, C118, C119 get 2.
3. The first five customers (C101 to C105) share no products with each other, so the printed matrix is 1.00 on the diagonal and 0.00 elsewhere. This is correct. Neighbours for these customers come from later customers.
4. The data is small: 60 purchases, 15 products. Most similarities are 0.17 to 0.50, so recommendations rest on few supporters. Treat this as a first version.

## Assumptions

- A customer's history is the set of distinct products bought. Repeat purchases and quantities do not add weight.
- Popularity counts distinct customers, not units sold.
- Customers whose every row is rejected are not in the engine. This does not occur in this file.
- Neighbour count is 3 (`k_neighbours`). `min_similarity` defaults to 0.0, and zero-similarity customers are excluded regardless.
- Product name and category come from the first valid row seen for each product.

## Structure

```
day4-recommendation-engine/
├── recommender.py
├── main.py
├── data/purchases.csv
├── output/recommendations.csv
├── tests/test_recommender.py
├── requirements.txt
└── README.md
```
