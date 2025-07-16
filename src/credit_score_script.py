import json
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def engineer_features(df):
    df['amount_usd'] = pd.to_numeric(df['actionData.amount'], errors='coerce') / 1e18 \
                       * pd.to_numeric(df['actionData.assetPriceUSD'], errors='coerce')
    df.dropna(subset=['amount_usd'], inplace=True)
    df['timestamp'] = pd.to_numeric(df['timestamp'])

    wallets = df.groupby('userWallet')
    
    features = pd.DataFrame(index=wallets.groups.keys())
    features['wallet_age_days'] = (wallets['timestamp'].max() - wallets['timestamp'].min()) / (60 * 60 * 24)
    features['tx_count'] = wallets['txHash'].nunique()
    features['tx_frequency'] = features['tx_count'] / (features['wallet_age_days'] + 1)
    features['total_deposited_usd'] = wallets.apply(lambda x: x[x['action'] == 'deposit']['amount_usd'].sum(), include_groups=False)
    features['total_borrowed_usd'] = wallets.apply(lambda x: x[x['action'] == 'borrow']['amount_usd'].sum(), include_groups=False)
    features['total_repaid_usd'] = wallets.apply(lambda x: x[x['action'] == 'repay']['amount_usd'].sum(), include_groups=False)
    features['liquidation_count'] = wallets.apply(lambda x: x[x['action'] == 'liquidationcall'].shape[0], include_groups=False)
    features['borrow_to_deposit_ratio'] = (features['total_borrowed_usd'] / (features['total_deposited_usd'] + 1e-6))
    features['repayment_ratio'] = (features['total_repaid_usd'] / (features['total_borrowed_usd'] + 1e-6))

    return features.fillna(0)

def phase_one_rule_based_scorer(features):
    scores = pd.DataFrame(index=features.index)
    scaler = MinMaxScaler()
    
    scores['css'] = scaler.fit_transform(features[['total_deposited_usd', 'wallet_age_days']]).mean(axis=1)
    
    rbs_features = pd.DataFrame(index=features.index)
    rbs_features['repayment'] = features['repayment_ratio']
    rbs_features['health'] = 1 - features['borrow_to_deposit_ratio']
    scores['rbs'] = scaler.fit_transform(rbs_features).mean(axis=1)
    
    scores['ras'] = 1.0
    scores.loc[features['liquidation_count'] > 0, 'ras'] = 0
    
    scores['pes'] = scaler.fit_transform(features[['tx_frequency', 'tx_count']]).mean(axis=1)
    
    weights = {'ras': 0.40, 'rbs': 0.30, 'css': 0.20, 'pes': 0.10}
    rule_based_score = (scores['ras'] * weights['ras'] + scores['rbs'] * weights['rbs'] + 
                        scores['css'] * weights['css'] + scores['pes'] * weights['pes'])
    
    return (rule_based_score * 1000).astype(int)

def phase_two_xgboost_model(features, target_scores):
    X = features
    y = target_scores
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    xgb_regressor = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.2,
        max_depth=3,
        colsample_bytree=1.0,
        random_state=42,
        n_jobs=-1        
    )

    xgb_regressor.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    predicted_scores = xgb_regressor.predict(X)
    scaler = MinMaxScaler(feature_range=(0, 1000))
    final_scores = scaler.fit_transform(predicted_scores.reshape(-1, 1)).astype(int)

    return final_scores.flatten()

def generate_hybrid_scores_from_json(file_path):
    print("Initiating credit scoring process...")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    df = pd.json_normalize(data)
    print(f"Loaded and parsed {len(df)} records from '{file_path}'.")

    engineered_features = engineer_features(df)
    if engineered_features.empty:
        raise ValueError("No valid transaction data found to process after engineering features.")
    print(f"Engineered features for {len(engineered_features)} unique wallets.")

    rule_based_scores = phase_one_rule_based_scorer(engineered_features)
    print("Completed Phase 1: Rule-based scores generated.")

    final_credit_scores = phase_two_xgboost_model(engineered_features, rule_based_scores)
    print("Completed Phase 2: XGBoost model trained and final scores generated.")

    results_df = pd.DataFrame({
        'rule_based_score': rule_based_scores,
        'credit_score': final_credit_scores
    }, index=engineered_features.index)

    output = results_df.reset_index().rename(columns={'index': 'userWallet'})
    print("Credit scoring process complete.")
    return output.to_dict('records')

def main():
    if len(sys.argv) != 2:
        print("Usage: python credit_score_script.py <input_json_file>")
        sys.exit(1)

    input_path = sys.argv[1]
    
    try:
        scores = generate_hybrid_scores_from_json(input_path)
        
        output_filename = 'wallet_credit_scores.json'
        with open(output_filename, 'w') as f:
            json.dump(scores, f, indent=2)
        print(f"\nProcess finished successfully. Results saved to '{output_filename}'.")
    
    except FileNotFoundError:
        print(f"\nError: The file was not found at the specified path: {input_path}")
        sys.exit(1)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\nError: Failed to process data. The file may be malformed or data processing failed.\nDetails: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()