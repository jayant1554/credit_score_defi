
# 🧠 Credit Score DeFi

This project builds a credit scoring system using on-chain DeFi transaction data (Aave). It assigns a **credit score (0–1000)** to wallets using:

- Rule-based heuristics
- Machine Learning predictions

---

## 📁 Project Structure

```
credit_score_defi/
├── raw_data/
│   └── aave_wallet_transactions.json     # Input data used for generating scores
├── src/
│   ├── credit_score_script.py            # Main script to process data and generate scores
│   ├── wallet_credit_scores.json         # Output file with scores for each wallet
│   ├── analysis.ipynb                    # Jupyter notebook with visualization and analysis
│   ├── analysis.md                       # Markdown summary of analysis.ipynb
│   ├── score_distribution.png            # Bar chart comparing rule vs ML scores
│   ├── joint_distribution.png            # Joint hex plot of scores
└── README.md
```

---

## 📥 Raw Input Data

### 📄 File: `raw_data/aave_wallet_transactions.json`

This file contains historical Aave V2 transaction data on the Polygon network. It includes:
- Wallet addresses
- Event types (e.g., deposit, borrow, repay, liquidation)
- Token details and transaction amounts
- Timestamps for behavioral analysis

This data is used to extract behavioral features and assign credit scores.

---

## 🚀 How to Run

### 🧪 1. Run the Scoring Script

```bash
cd 
python \sr\credit_score_script.py \raw_transaction_data\user-wallet-transactions.json    
```

This will:
- Load the raw data from `../raw_data/aave_wallet_transactions.json`
- Process and evaluate each wallet
- Output:
  - `wallet_credit_scores.json`
  - `score_distribution.png`
  - `joint_distribution.png`

---

### 📊 2. Explore Notebook (Optional)

```bash
jupyter notebook exp.ipynb
jupyter notebook analysis.ipynb
```

For interactive visual analysis and score comparison.

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `wallet_credit_scores.json` | Credit score output for each wallet |
| `score_distribution.png` | Bar chart of score ranges |
| `joint_distribution.png` | Joint plot: Rule vs ML score correlation |
| `analysis.md` | Summary of notebook insights |

---

## 👤 Author

**Jayant Bisht**  
GitHub: [@jayant1554](https://github.com/jayant1554)
