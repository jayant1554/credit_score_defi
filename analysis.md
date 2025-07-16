# Analysis of Wallet Credit Scores

This document presents a detailed analysis of wallet credit scores derived from on-chain data. The analysis compares two scoring models: a **Rule-Based Model** and a **Machine Learning (ML)-Based Model**. The primary goal is to understand the distribution of scores, identify discrepancies between the two models, and classify user behavior based on these scores.

The data is sourced from `wallet_credit_scores.json` and processed using Python libraries such as Pandas, Matplotlib, and Seaborn.

---

## 1. Score Distribution Analysis

The first step in the analysis is to understand how credit scores are distributed across the user base for both models. This provides a high-level overview of the financial health and on-chain activity of the wallets.

### 1.1. Pie Chart Representation

The pie charts below illustrate the percentage of wallets falling into different score ranges (0-99, 100-199, etc.).

![Score Distribution Pie Charts](https://github.com/jayant1554/credit_score_defi/blob/main/analys_imag/piechartdf.png)

**Key Observations:**

* **Rule-Based Model:**
    * A staggering **83%** of wallets are concentrated in the **500-599** score range.
    * This suggests that the rule-based model has low variance and tends to group most users into a single, narrow category, making it difficult to differentiate between them.
    * Only a tiny fraction of wallets score outside this range, indicating a lack of granularity in the scoring logic.
* **ML-Based Model:**
    * The distribution is more balanced and spread out, with significant portions of users in the **100-199 (14.1%)**, **700-799 (26.6%)**, and **800-899 (25.1%)** ranges.
    * This model provides a much better differentiation between wallets, identifying distinct user segments from low to high reliability.
    * The presence of a large group in the higher score ranges (700+) suggests that many users exhibit behavior indicative of high creditworthiness.

### 1.2. Bar Chart Comparison

The bar chart provides a side-by-side comparison of the number of wallets in each score range for the two models.

![Score Range Bar Chart])

**Key Observations:**

* This chart reinforces the findings from the pie charts. The rule-based model's overwhelming concentration in the 100-199 range is clearly visible, while the ML-based model shows a more realistic and useful distribution.
* The ML-based model successfully identifies users across the spectrum, which is essential for any practical application like targeted services or risk assessment.

---

## 2. Model Discrepancy Analysis

To further understand the differences between the two models, we analyzed the score differential (`ML Score - Rule-Based Score`).

### 2.1. Kernel Density Estimate (KDE) Plot

The KDE plot shows the distribution of the score differences.

![KDE Plot of Score Differences](https://github.com/jayant1554/credit_score_defi/blob/main/analys_imag/kdedf.png)

**Statistical Summary:**

* **Mean Difference:** 179.07
* **Standard Deviation:** 54.59
* **Min Difference:** -151
* **Max Difference:** 300

**Key Observations:**

* The distribution is heavily skewed to the right, with a strong peak around a score difference of **177**. This indicates that the ML-based model consistently scores wallets **significantly higher** than the rule-based model.
* The **mean difference of 179** confirms this trend. The ML model appears to recognize positive on-chain behaviors that the rule-based model either ignores or penalizes.
* The minimum difference of -151 shows that in some rare cases, the rule-based model scored a wallet higher than the ML model. These outliers could be wallets engaging in specific activities that trigger a high score in the rule-based system but are flagged as neutral or risky by the ML model.

### 2.2. Confusion Matrix of Behavior Classifications

To compare the models' classifications, we categorized wallets based on their scores and created a confusion matrix.

**Classification Categories:**
* **Highly Reliable:** Score >= 800
* **Reliable:** Score >= 650
* **Risky / Bot-like:** Score < 650

| ML Classification | Rule-Based: Reliable | Rule-Based: Risky / Bot-like | All |
| :--- | :--- | :--- | :--- |
| **Highly Reliable** | 6 | 494 | 500 |
| **Reliable** | 0 | 2895 | 2895 |
| **Risky / Bot-like**| 0 | 102 | 102 |
| **All** | 6 | 3491 | 3497 |

**Key Observations:**

* The rule-based model classifies almost all wallets (**3491 out of 3497**) as "Risky / Bot-like." This is a major flaw, as it fails to identify creditworthy users.
* The ML model, in contrast, identifies **500 "Highly Reliable"** and **2895 "Reliable"** users who were misclassified by the rule-based model.
* There is a significant disagreement between the two models. The ML model provides a more optimistic and likely more accurate assessment of user behavior by capturing a wider range of on-chain activities.

---

## 3. User Behavior Classification & Profiles

Based on the superior granularity of the ML-based model, we can classify user behavior with greater confidence and create distinct user profiles.

### 3.1. Wallet Classification by Behavior Type

The pie chart below shows the breakdown of user types according to the ML model's scores.

![User Behavior Pie Chart](https://github.com/jayant1554/credit_score_defi/blob/main/analys_imag/userbehavdf.png)

* **Reliable (82.8%):** The vast majority of users fall into this category. These are likely regular users with a consistent but not necessarily extensive history of on-chain activity. They represent a stable and predictable user base.
* **Highly Reliable (14.3%):** This segment represents the "power users" of the ecosystem. Their high scores suggest a long, diverse, and positive on-chain history.
* **Risky / Bot-like (2.9%):** A small percentage of users are classified as risky. This group may include new wallets, wallets with very little history, or those engaging in high-risk activities.

### 3.2. Wallet Profile Comparison

#### **Highly Reliable (Score ≥ 800)**
These are the **"DeFi Power Users"** or **"Whales"**. They are highly active and deeply integrated into the on-chain economy.
* **On-Chain History:** Long wallet age with a rich and extensive transaction history.
* **Financial Activity:** High transaction volume and a substantial, diverse portfolio of assets (e.g., ETH, blue-chip DeFi tokens, stablecoins).
* **DeFi Engagement:** Interacts with a wide array of protocols, including lending platforms (Aave, Compound), decentralized exchanges (Uniswap, Curve), and yield farming/staking protocols.
* **Credit History:** Likely has a proven history of borrowing and, crucially, repaying loans. They often maintain a healthy loan-to-value (LTV) ratio and avoid liquidations.
* **Governance:** May hold governance tokens and actively participate in protocol voting, indicating a long-term interest in the ecosystem's health.

#### **Reliable (Score 650 - 799)**
This group consists of **"Regular Users"** or **"Retail Participants"**. They are the backbone of the DeFi ecosystem, showing consistent and predictable behavior.
* **On-Chain History:** Moderate wallet age with a steady transaction history.
* **Financial Activity:** Holds a balanced portfolio, perhaps with a focus on major assets like ETH and stablecoins. Transaction volumes are moderate.
* **DeFi Engagement:** Interacts with a few well-known, reputable protocols. They might use a DEX for swapping tokens and a lending platform for earning interest but are less likely to engage in complex yield farming strategies.
* **Credit History:** May have taken out small, over-collateralized loans and have a clean repayment record.

#### **Risky / Bot-like (Score < 650)**
This category includes **"New Users," "Speculators,"** or **"Bots"**. Their on-chain footprint is either too small to be reliable or exhibits patterns that are flagged as high-risk.
* **On-Chain History:** Very recent wallet age with minimal or erratic transaction history.
* **Financial Activity:** Low asset values or a portfolio concentrated in high-risk, volatile "meme coins."
* **DeFi Engagement:** Limited to no interaction with established DeFi protocols. Activity might be confined to a single DEX or a new, unaudited application.
* **Credit History:** No history of borrowing or lending on major platforms.
* **Bot-like Patterns:** May show automated, repetitive transaction patterns, such as frequent, small-value swaps characteristic of MEV bots.

---

## 4. Conclusion

The analysis clearly demonstrates the superiority of the **ML-based model** over the rule-based model for assessing wallet creditworthiness.

* The **rule-based model** is too simplistic and lacks the nuance to differentiate between users, incorrectly flagging the majority as risky.
* The **ML-based model** provides a well-distributed and granular scoring system that effectively segments users into meaningful categories. It consistently identifies creditworthy behavior that the rule-based model overlooks.

This analysis validates the use of machine learning for on-chain credit scoring and provides a solid foundation for developing more sophisticated financial products and risk management strategies in the DeFi ecosystem.
