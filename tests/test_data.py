import unittest

import pandas as pd

from src.data import (
    OFFICIAL_DATA_PATH,
    calculate_summary,
    filter_transactions,
    load_transactions,
    monthly_median_trend,
    preprocess_transactions,
    resale_price_distribution,
    valid_flat_types,
    valid_towns,
)


class TransactionDataTests(unittest.TestCase):
    def test_preprocessing_discards_malformed_rows(self):
        raw = pd.DataFrame(
            [
                {
                    "month": "2025-01",
                    "town": " bedok ",
                    "flat_type": "4 room",
                    "block": "1",
                    "street_name": "BEDOK RD",
                    "storey_range": "01 TO 03",
                    "floor_area_sqm": "90",
                    "flat_model": "Model A",
                    "lease_commence_date": "1980",
                    "resale_price": "500000",
                },
                {
                    "month": "not-a-month",
                    "town": "BEDOK",
                    "flat_type": "4 ROOM",
                    "block": "2",
                    "street_name": "BEDOK RD",
                    "storey_range": "01 TO 03",
                    "floor_area_sqm": "90",
                    "flat_model": "Model A",
                    "lease_commence_date": "1980",
                    "resale_price": "510000",
                },
                {
                    "month": "2025-01",
                    "town": "BEDOK",
                    "flat_type": "4 ROOM",
                    "block": "3",
                    "street_name": "BEDOK RD",
                    "storey_range": "01 TO 03",
                    "floor_area_sqm": "invalid",
                    "flat_model": "Model A",
                    "lease_commence_date": "1980",
                    "resale_price": "520000",
                },
            ]
        )

        cleaned, rows_dropped = preprocess_transactions(raw)

        self.assertEqual(len(cleaned), 1)
        self.assertEqual(rows_dropped, 2)
        self.assertEqual(cleaned.iloc[0]["town"], "BEDOK")
        self.assertEqual(cleaned.iloc[0]["flat_type"], "4 ROOM")
        self.assertEqual(cleaned.iloc[0]["resale_price"], 500_000)

    @unittest.skipUnless(OFFICIAL_DATA_PATH.exists(), "Official dataset is not available")
    def test_official_dataset_filters_and_aggregates(self):
        result = load_transactions()
        self.assertFalse(result.is_demo)
        self.assertGreater(len(result.data), 200_000)
        self.assertIn("BEDOK", valid_towns(result.data))
        self.assertIn("4 ROOM", valid_flat_types(result.data, "BEDOK"))

        filtered = filter_transactions(
            result.data,
            "BEDOK",
            "4 ROOM",
            pd.Timestamp("2024-01-01").date(),
            pd.Timestamp("2024-12-01").date(),
        )
        self.assertFalse(filtered.empty)
        summary = calculate_summary(filtered, 600_000)
        self.assertEqual(summary.transaction_count, len(filtered))
        self.assertGreaterEqual(summary.within_budget_percentage, 0)
        self.assertLessEqual(summary.within_budget_percentage, 100)
        self.assertEqual(monthly_median_trend(filtered)["period"].is_monotonic_increasing, True)
        self.assertEqual(resale_price_distribution(filtered)["transactions"].sum(), len(filtered))


if __name__ == "__main__":
    unittest.main()
