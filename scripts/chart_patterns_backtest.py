"""Chart Patterns Backtest — ML with vs without chart pattern features."""

import time
from trading_engine.config import WATCHLIST
from trading_engine.price_engine import get_history
from trading_engine.ml_model import MLSignalGenerator


def main():
    print("=" * 120)
    print("  CHART PATTERNS IMPACT — ML with Support/Resistance, Double Top/Bottom, Fibonacci, Trendlines")
    print("=" * 120)

    print("\n  Downloading data...")
    cached = {}
    for sym in WATCHLIST:
        df = get_history(sym, days=600)
        if not df.empty:
            cached[sym] = df
            print(f"    {sym}: {len(df)} days")

    print("\n  Building features with chart patterns (this may take 30-60 seconds)...")
    t0 = time.time()

    configs = [
        {"capital": 1000, "conf": 0.65, "label": "$1K / 65%"},
        {"capital": 10000, "conf": 0.60, "label": "$10K / 60%"},
        {"capital": 10000, "conf": 0.65, "label": "$10K / 65%"},
    ]

    print(f"\n  {'Config':<15} {'Trades':>7} {'Closed':>7} {'Win%':>6} "
          f"{'Net P&L':>10} {'Return':>8} {'MaxDD':>7}")
    print(f"  {'-' * 65}")

    for cfg in configs:
        ml = MLSignalGenerator(
            train_window=200, min_train=80,
            confidence_threshold=cfg["conf"],
        )
        r = ml.train_and_backtest(
            history_data=cached,
            initial_capital=cfg["capital"],
            max_position_pct=0.15,
            stop_loss_pct=0.03,
            take_profit_pct=0.05,
        )

        if "error" in r:
            print(f"  {cfg['label']:<15} ERROR: {r['error']}")
            continue

        daily = r.get("daily_results", [])
        max_v = max(d["total_value"] for d in daily) if daily else cfg["capital"]
        min_v = min(d["total_value"] for d in daily) if daily else cfg["capital"]
        dd = (max_v - min_v) / max_v * 100 if max_v > 0 else 0

        icon = "\U0001f7e2" if r["total_pnl_net"] > 0 else "\U0001f534"
        print(f"  {cfg['label']:<15} {r['total_trades']:>7} {r['closed_trades']:>7} "
              f"{r['win_rate']:>5.1f}% {icon} ${r['total_pnl_net']:>+8.2f} "
              f"{r['return_pct']:>+7.2f}% {dd:>6.2f}%")

    elapsed = time.time() - t0
    print(f"\n  Completed in {elapsed:.0f}s")

    # Feature importance
    print(f"\n  Training final model to show feature importance...")
    ml = MLSignalGenerator(train_window=200, min_train=80, confidence_threshold=0.65)

    from trading_engine.ml_model import build_feature_matrix
    import pandas as pd
    import lightgbm as lgb

    all_features = {}
    spy_features = None
    if "SPY" in cached:
        spy_feat = build_feature_matrix(cached["SPY"])
        spy_features = spy_feat[["rsi_2", "rsi_14", "ibs", "return_1d",
                                 "return_5d", "volatility_5d",
                                 "close_vs_sma_200"]].copy()
        spy_features.columns = [f"spy_{c}" for c in spy_features.columns]

    for sym, df in cached.items():
        feat = build_feature_matrix(df)
        if spy_features is not None and sym != "SPY":
            feat = feat.join(spy_features, how="left")
        feat = feat.dropna()
        if len(feat) > 80:
            all_features[sym] = feat

    X_all = []
    y_all = []
    feature_cols = None
    for sym, feat in all_features.items():
        if feature_cols is None:
            feature_cols = [c for c in feat.columns if c not in {"target", "target_dir"}]
        valid = feat.dropna(subset=feature_cols + ["target_dir"])
        if len(valid) > 30:
            X_all.append(valid[feature_cols])
            y_all.append(valid["target_dir"])

    if X_all:
        X = pd.concat(X_all)
        y = pd.concat(y_all)
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.7, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42,
        )
        model.fit(X, y)

        importance = dict(zip(feature_cols, model.feature_importances_))
        sorted_imp = sorted(importance.items(), key=lambda x: -x[1])

        # Highlight chart pattern features
        chart_features = {
            "dist_to_resistance_pct", "resistance_strength",
            "dist_to_support_pct", "support_strength",
            "double_top", "double_bottom",
            "head_shoulders", "inv_head_shoulders",
            "dist_to_fib_support_pct", "dist_to_fib_resistance_pct",
            "trend_slope_pct", "channel_position", "channel_width_pct",
            "near_channel_upper", "near_channel_lower",
        }

        print(f"\n  {'=' * 80}")
        print(f"  FEATURE IMPORTANCE (Top 30)")
        print(f"  {'=' * 80}")
        print(f"  {'#':>3} {'Feature':<35} {'Importance':>10} {'Type':<15}")
        print(f"  {'-' * 70}")

        for rank, (feat_name, imp) in enumerate(sorted_imp[:30], 1):
            is_chart = feat_name in chart_features
            is_spy = feat_name.startswith("spy_")
            tag = "\U0001f4ca CHART" if is_chart else ("\U0001f30e SPY" if is_spy else "Technical")
            marker = " <<<" if is_chart else ""
            print(f"  {rank:>3}. {feat_name:<35} {imp:>10} {tag}{marker}")

        print(f"\n  Chart pattern features in top 30:")
        chart_in_top30 = [(r, n, i) for r, (n, i) in enumerate(sorted_imp[:30], 1) if n in chart_features]
        if chart_in_top30:
            for rank, name, imp in chart_in_top30:
                print(f"    #{rank}: {name} (importance: {imp})")
        else:
            print("    None in top 30")

        total_imp = sum(importance.values())
        chart_imp = sum(importance.get(f, 0) for f in chart_features)
        print(f"\n  Chart patterns share: {chart_imp/total_imp*100:.1f}% of total feature importance")


if __name__ == "__main__":
    main()
