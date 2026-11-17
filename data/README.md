# Data

The dataset is **not committed** to this repo. Download it yourself and place it at
`data/raw/marketing_campaign.csv`.

| | |
|---|---|
| Name | Customer Personality Analysis |
| Source | Kaggle: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis |
| License | CC0: Public Domain (as listed on Kaggle) |
| File | `marketing_campaign.csv` (tab-separated, despite the `.csv` extension) |
| Shape | 2,240 rows, 29 columns |

I ran the original analysis on a copy of this data supplied as course material. It has the same
2,240 rows and 29 columns as the public Kaggle file, but I have not checked the two byte for
byte. If `make reproduce` reports differences, that is the first thing to check.

## Download

With the Kaggle CLI (needs a free Kaggle account and API token):

```bash
kaggle datasets download -d imakash3011/customer-personality-analysis -p data/raw --unzip
```

Or download the zip from the Kaggle page and unzip `marketing_campaign.csv` into `data/raw/`.

## Columns used

- **Customer:** `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome`, `Dt_Customer`, `Recency`, `Complain`
- **Spending, last two years:** `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds`
- **Purchases by channel:** `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth`
- **Campaign responses:** `AcceptedCmp1`–`AcceptedCmp5`, `Response`
- **Ignored:** `ID` (identifier), `Z_CostContact` and `Z_Revenue` (constant)

## No data? Use the synthetic generator

`make demo` and the tests use `segmentation.synthetic.make_customers()`, which generates fake
customers with the same schema and the same kinds of data problems. Its numbers mean nothing
about real customers; it only proves the code runs.
