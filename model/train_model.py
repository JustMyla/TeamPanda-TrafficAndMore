import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
from pathlib import Path

current_script_path = Path(__file__).resolve()
project_root = current_script_path.parent.parent
sys.path.append(str(project_root))
from pipeline import prepare_data, markeer_omslagpunten, undersample

def train_and_save_lgbm():
    file_path = project_root / "data" / "export_location_modeling" / "export_location_modeling.csv"
    df_raw = pd.read_csv(file_path)

    print("[1/5] Running pipeline...")
    df = prepare_data(df_raw)
    df = markeer_omslagpunten(df)

    print("[2/5] Preparing features...")
    features = ['gem_intensiteit_smooth', 'tijd', 'day_of_week']
    target = 'file_omslag_flag'
    df_clean = df.dropna(subset=features + [target])
    X, y = df_clean[features], df_clean[target]

    print("[3/5] Splitting and balancing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_df_balanced = undersample(pd.concat([X_train, y_train], axis=1), minority_class=1, ratio=3, random=42)
    X_train_balanced, y_train_balanced = train_df_balanced[features], train_df_balanced[target]

    print("[4/5] Training model...")
    model = lgb.LGBMClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        random_state=42,
        verbose=-1
    )
    model.fit(X_train_balanced, y_train_balanced)

    print("[5/5] Evaluating and saving...")
    y_pred = model.predict(X_test)
    print("\n" + classification_report(y_test, y_pred, target_names=['Geen omslag', 'Omslag']))

    output_path = project_root / 'model' / 'lightgbm_traffic_model.pkl'
    joblib.dump(model, output_path)
    print(f"\n✓ Model saved: {output_path}")

if __name__ == "__main__":
    train_and_save_lgbm()