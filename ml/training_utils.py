"""Global Cross-Sectional Panel training engine with GPU/CPU parallelism, OOF variance calibration,
multi-seed bagging, IC-aware loss, and institutional quant evaluation.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
import yfinance as yf

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from backend.config import (  # noqa: E402
    BASE_ALGORITHMS,
    LOOKBACK_WINDOW,
    METRICS_SUMMARY_PATH,
    SECTORS,
    SORTED_TICKERS,
    STACKING_BASE_ALGORITHMS,
    START_DATE,
)
from backend.services.attention_net import (  # noqa: E402
    TemporalAttentionNet,
    get_torch_device,
    save_attention_net_bundle,
)
from backend.services.feature_engineering import (  # noqa: E402
    FEATURE_COLUMNS,
    LONG_HORIZON_CANDIDATE_COLUMNS,
    build_panel_dataframe,
    fetch_macro_context,
)


def configure_training_environment(models_dir):
    """Configure CUDA Tensor Cores, deterministic seeds, and output paths."""
    warnings.filterwarnings("ignore")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(os.path.dirname(METRICS_SUMMARY_PATH), exist_ok=True)

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = get_torch_device()
    cpu_count = os.cpu_count() or 8
    gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "None (CPU fallback)"
    print(f"=== Global Panel Trainer | GPU: {gpu_name} | CPU Threads: {cpu_count} ===")
    return device


def parallel_download_universe(stock_map, start_date=START_DATE, max_workers=10):
    """Download 20+ years of daily OHLCV safely for all NSE stocks using bulk download + sequential retry."""
    end_date = datetime.now().strftime("%Y-%m-%d")
    tickers = sorted(set(stock_map.values()))
    results = {}

    print(f"Downloading full OHLCV history ({start_date} -> {end_date}) for {len(stock_map)} NSE stocks...")
    try:
        bulk = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        if bulk is not None and not bulk.empty:
            if isinstance(bulk.columns, pd.MultiIndex):
                lvl0 = set(bulk.columns.get_level_values(0))
                for t in tickers:
                    if t in lvl0:
                        sub = bulk[t].dropna(how="all")
                        if not sub.empty:
                            results[t] = sub
            elif len(tickers) == 1:
                sub = bulk.dropna(how="all")
                if not sub.empty:
                    results[tickers[0]] = sub
    except Exception as exc:
        print(f"Warning: bulk download encountered an error ({exc}); falling back to per-ticker fetch.")

    # Sequentially retry any missing tickers using isolated yf.Ticker().history (zero yf.shared race)
    missing = [(name, t) for name, t in stock_map.items() if t not in results]
    for name, ticker in missing:
        for attempt in range(2):
            try:
                df = yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=True)
                if df is not None and not df.empty:
                    results[ticker] = df
                    break
            except Exception as exc:
                if attempt == 1:
                    print(f"Warning: failed to download {name} ({ticker}): {exc}")

    print(f"Downloaded {len(results)}/{len(stock_map)} stocks successfully.")
    return results


def build_global_training_panel(stock_map, start_date=START_DATE):
    """Download universe + macro context and assemble the cross-sectional panel DataFrame."""
    macro_df = fetch_macro_context(start_date=start_date, end_date=datetime.now(), use_cache=False)
    raw_map = parallel_download_universe(stock_map, start_date=start_date)
    panel = build_panel_dataframe(raw_map, macro_df=macro_df)
    print(
        f"Global Panel ready: {len(panel):,} rows | {panel['ticker'].nunique()} stocks | "
        f"{panel['date'].min().strftime('%Y-%m-%d')} -> {panel['date'].max().strftime('%Y-%m-%d')} | "
        f"{len(FEATURE_COLUMNS)} features"
    )
    return panel


def attach_horizon_targets(panel_df, horizon):
    """Attach strictly leak-free forward log returns, market-neutral alpha target_z, and total raw_target_z."""
    df = panel_df.sort_values(["ticker", "date"]).copy()
    grouped_close = df.groupby("ticker")["Close"]
    df["fwd_ret"] = np.log(grouped_close.shift(-horizon) / df["Close"])
    df["vol_scale"] = df["vol_21d"].clip(lower=0.006, upper=0.10) * np.sqrt(float(horizon))

    # Same-date cross-sectional median return separates unlearnable overnight market shock (beta)
    # from learnable stock-specific relative outperformance (cross-sectional alpha).
    df["mkt_fwd_ret"] = df.groupby("date")["fwd_ret"].transform("median")
    df["excess_fwd_ret"] = df["fwd_ret"] - df["mkt_fwd_ret"]

    # Stricter outlier winsorization & exact per-date zero-centering for multi-day cross-sectional alpha
    z_clip = 3.5 if horizon == 1 else 2.5
    raw_alpha_z = (df["excess_fwd_ret"] / df["vol_scale"]).clip(-z_clip, z_clip)
    df["target_z"] = raw_alpha_z - raw_alpha_z.groupby(df["date"]).transform("mean")
    # Total return z-score preserved for P10/P50/P90 uncertainty quantile cones
    df["raw_target_z"] = (df["fwd_ret"] / df["vol_scale"]).clip(-3.5, 3.5)

    # Track per-ticker local row position for zero-copy sequence slicing before dropping tail NaNs
    df["local_pos"] = df.groupby("ticker").cumcount()

    # Keep full feature index for sequence lookback, mark rows with valid forward target
    df["has_target"] = df["fwd_ret"].notna() & df["target_z"].notna() & df["raw_target_z"].notna()
    df.sort_values(["date", "ticker"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _get_horizon_profile(horizon):
    """Return horizon-regime configuration (preserves exact h=1d behavior while hardening h>=5d)."""
    if horizon == 1:
        return {
            "use_entity_ids": True,
            "half_life_years": 5.0,
            "max_features": 68,
            "stride": 1,
            "cv_seeds": (42, 102),
            "prod_seeds": (42, 102, 202),
            "cv_attn_epochs": 20,
            "prod_attn_epochs": 28,
        }
    if horizon <= 10:
        return {
            "use_entity_ids": False,
            "half_life_years": 12.0,
            "max_features": 32,
            "stride": 2 if horizon == 5 else 3,
            "cv_seeds": (42, 102, 202),
            "prod_seeds": (42, 102, 202, 302),
            "cv_attn_epochs": 18,
            "prod_attn_epochs": 24,
        }
    return {
        "use_entity_ids": False,
        "half_life_years": 25.0,
        "max_features": 24,
        "stride": 5 if horizon == 21 else 10,
        "cv_seeds": (42, 102, 202),
        "prod_seeds": (42, 102, 202, 302, 402),
        "cv_attn_epochs": 18,
        "prod_attn_epochs": 24,
    }


def compute_recency_weights(dates, half_life_years=5.0):
    """Exponential recency sample weighting so models learn multi-cycle structure while favoring current regimes."""
    max_date = dates.max()
    age_years = (max_date - dates).dt.days.values.astype(np.float64) / 365.25
    weights = np.exp(-np.log(2.0) * age_years / half_life_years)
    return weights / np.mean(weights)


def purged_date_splits(unique_dates, n_splits=5, embargo_days=2, min_train_frac=0.35):
    """Generate date-level expanding walk-forward (train_dates, val_dates) with strict embargo purging."""
    n_dates = len(unique_dates)
    min_train = max(LOOKBACK_WINDOW + 120, int(n_dates * min_train_frac))
    remaining = n_dates - min_train
    fold_size = max(40, remaining // n_splits)

    splits = []
    for fold in range(n_splits):
        val_start = min_train + fold * fold_size
        val_end = n_dates if fold == n_splits - 1 else min(n_dates, val_start + fold_size)
        train_end = val_start - embargo_days
        if train_end <= LOOKBACK_WINDOW + 60 or val_end - val_start < 20:
            continue
        splits.append((unique_dates[:train_end], unique_dates[val_start:val_end]))
    return splits


def select_stable_features(
    x_dev, y_dev, max_features=68, dates_dev=None, candidate_indices=None, horizon=1
):
    """Screen alpha features by cross-sectional daily IC stability strictly inside Dev (zero test leakage)."""
    stds = np.std(x_dev, axis=0)
    if candidate_indices is not None:
        cand_set = set(int(c) for c in candidate_indices)
        valid_cols = np.array([c for c in range(x_dev.shape[1]) if c in cand_set and stds[c] > 1e-6], dtype=int)
    else:
        valid_cols = np.where(stds > 1e-6)[0]

    if len(valid_cols) <= max_features:
        return valid_cols.tolist(), []

    if horizon == 1 or dates_dev is None:
        chunks = np.array_split(np.arange(len(y_dev)), 4)
        scores = []
        for col in valid_cols:
            chunk_ics = []
            for idx in chunks:
                xc = x_dev[idx, col]
                yc = y_dev[idx]
                if np.std(xc) < 1e-8 or np.std(yc) < 1e-8:
                    chunk_ics.append(0.0)
                else:
                    chunk_ics.append(float(np.corrcoef(xc, yc)[0, 1]))
            mean_abs_ic = float(np.mean(np.abs(chunk_ics)))
            sign_consistency = abs(float(np.mean(np.sign(chunk_ics))))
            scores.append((int(col), mean_abs_ic * (0.5 + 0.5 * sign_consistency)))

        scores.sort(key=lambda item: item[1], reverse=True)
        selected = sorted([col for col, _ in scores[:max_features]])
        return selected, []

    # Long-horizon SOTA screening: non-overlapping sampled cross-sectional daily IC across 4 eras
    date_frame = pd.DataFrame({"date": dates_dev, "row": np.arange(len(dates_dev))})
    date_to_rows = {d: grp["row"].values for d, grp in date_frame.groupby("date")}
    unique_dates = np.sort(np.array(list(date_to_rows.keys())))
    era_dates_list = np.array_split(unique_dates, 4)
    step = max(1, horizon // 3)

    era_means = []
    for era_dates in era_dates_list:
        sampled_dates = era_dates[::step]
        daily_ics = []
        for d in sampled_dates:
            r_idx = date_to_rows[d]
            if len(r_idx) < 12:
                continue
            y_d = y_dev[r_idx]
            yc = y_d - np.mean(y_d)
            y_ss = float(np.sum(yc ** 2))
            if y_ss < 1e-10:
                continue
            x_d = x_dev[r_idx][:, valid_cols]
            xc = x_d - np.mean(x_d, axis=0, keepdims=True)
            x_ss = np.sum(xc ** 2, axis=0)
            denom = np.sqrt(x_ss * y_ss)
            cov = np.sum(xc * yc[:, None], axis=0)
            ic_d = np.where(denom > 1e-12, cov / np.maximum(denom, 1e-12), 0.0)
            daily_ics.append(ic_d)
        if daily_ics:
            era_means.append(np.mean(daily_ics, axis=0))

    if not era_means:
        return valid_cols[:max_features].tolist(), []

    era_mat = np.vstack(era_means)  # shape: (n_eras, len(valid_cols))
    mean_ic = np.mean(era_mat, axis=0)
    std_ic = np.std(era_mat, axis=0)
    sign_cons = np.abs(np.mean(np.sign(era_mat), axis=0))
    stability_scores = (np.abs(mean_ic) / (std_ic + 0.015)) * (0.25 + 0.75 * (sign_cons ** 2))

    ranked = sorted(
        [
            (
                int(valid_cols[i]),
                float(stability_scores[i]),
                float(mean_ic[i]),
                float(std_ic[i]),
                float(sign_cons[i]),
            )
            for i in range(len(valid_cols))
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    selected = sorted([item[0] for item in ranked[:max_features]])
    top_diag = [(item[0], item[2], item[3], item[4]) for item in ranked[:8]]
    return selected, top_diag


class PanelSequenceDataset(Dataset):
    """Zero-copy memory-efficient PyTorch Dataset for 150,000+ cross-sectional stock sequences."""

    def __init__(self, stock_arrays, row_meta, seq_len=LOOKBACK_WINDOW, use_entity_ids=True):
        """
        Args:
            stock_arrays: dict[stock_id -> np.ndarray of shape (T_i, F) float32]
            row_meta: list of tuples (stock_id, local_pos, sector_id, target_z, sample_weight, panel_row_idx, date_code)
        """
        self.stock_arrays = stock_arrays
        self.row_meta = row_meta
        self.seq_len = seq_len
        self.use_entity_ids = use_entity_ids
        self.neutral_stock_id = len(SORTED_TICKERS) + 2
        self.neutral_sector_id = len(SECTORS) + 2

    def __len__(self):
        return len(self.row_meta)

    def __getitem__(self, idx):
        stock_id, local_pos, sector_id, target_z, weight, _, date_code = self.row_meta[idx]
        seq = self.stock_arrays[stock_id][local_pos - self.seq_len + 1 : local_pos + 1]
        out_sid = stock_id if self.use_entity_ids else self.neutral_stock_id
        out_sec = sector_id if self.use_entity_ids else self.neutral_sector_id
        return (
            torch.from_numpy(seq),
            torch.tensor(out_sid, dtype=torch.long),
            torch.tensor(out_sec, dtype=torch.long),
            torch.tensor(target_z, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
            torch.tensor(date_code, dtype=torch.long),
        )


def _build_stock_sequence_store(df_all, scaled_selected):
    """Build compact per-stock 2D float32 arrays (uses < 100 MB RAM for 180k rows)."""
    stock_arrays = {}
    # Order by ticker + local_pos so local_pos indexes directly into stock_arrays[stock_id]
    order_df = pd.DataFrame(
        {
            "row_idx": np.arange(len(df_all)),
            "stock_id": df_all["stock_id"].values,
            "local_pos": df_all["local_pos"].values,
        }
    ).sort_values(["stock_id", "local_pos"])

    for sid, grp in order_df.groupby("stock_id"):
        stock_arrays[int(sid)] = scaled_selected[grp["row_idx"].values].astype(np.float32)
    return stock_arrays


def _make_sequence_meta(df_all, row_indices, weights_all, date_codes_all=None):
    """Create row metadata tuples including date_code for indices with >= LOOKBACK_WINDOW prior history."""
    sids = df_all["stock_id"].values
    lpos = df_all["local_pos"].values
    secs = df_all["sector_id"].values
    ys = df_all["target_z"].values

    meta = []
    for ridx in row_indices:
        lp = int(lpos[ridx])
        if lp >= LOOKBACK_WINDOW - 1:
            dcode = int(date_codes_all[ridx]) if date_codes_all is not None else 0
            meta.append(
                (
                    int(sids[ridx]),
                    lp,
                    int(secs[ridx]),
                    float(ys[ridx]) if np.isfinite(ys[ridx]) else 0.0,
                    float(weights_all[ridx]),
                    int(ridx),
                    dcode,
                )
            )
    return meta


class DateGroupedBatchSampler(Sampler):
    """Yields multi-date cross-sectional batches rotated over stride offsets each epoch."""

    def __init__(self, date_codes, dates_per_batch=4, stride=1, seed=42):
        self.dates_per_batch = max(1, int(dates_per_batch))
        self.stride = max(1, int(stride))
        self.seed = int(seed)
        self._epoch = 0

        grouped = {}
        for idx, d in enumerate(date_codes):
            grouped.setdefault(int(d), []).append(idx)
        self.date_to_indices = {d: idxs for d, idxs in grouped.items() if len(idxs) >= 10}
        self.all_dates = sorted(self.date_to_indices.keys())

    def _epoch_dates(self, offset):
        if self.stride > 1:
            chosen = [d for d in self.all_dates if (d % self.stride) == offset]
            if len(chosen) >= 4:
                return chosen
        return list(self.all_dates)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        dates = self._epoch_dates(self._epoch % self.stride)
        self._epoch += 1
        rng.shuffle(dates)

        for start in range(0, len(dates), self.dates_per_batch):
            batch = []
            for d in dates[start : start + self.dates_per_batch]:
                batch.extend(self.date_to_indices[d])
            if batch:
                yield batch

    def __len__(self):
        approx_dates = max(1, len(self.all_dates) // self.stride)
        return max(1, (approx_dates + self.dates_per_batch - 1) // self.dates_per_batch)


def _huber_ic_loss(preds, targets, weights, date_codes=None, delta=1.5, ic_weight=0.05):
    """Weighted Huber loss + same-date cross-sectional (or mini-batch) Pearson IC objective."""
    diff = torch.abs(preds - targets)
    huber = torch.where(diff <= delta, 0.5 * (diff ** 2), delta * (diff - 0.5 * delta))
    weighted_huber = torch.sum(huber * weights) / (torch.sum(weights) + 1e-8)

    if ic_weight <= 0.0:
        return weighted_huber

    if date_codes is not None and date_codes.numel() > 10:
        date_ics = []
        for d in torch.unique(date_codes):
            mask = date_codes == d
            if int(mask.sum()) < 10:
                continue
            p_d = preds[mask]
            t_d = targets[mask]
            p_c = p_d - p_d.mean()
            t_c = t_d - t_d.mean()
            denom = torch.sqrt(torch.sum(p_c ** 2) * torch.sum(t_c ** 2) + 1e-8)
            date_ics.append(torch.sum(p_c * t_c) / denom)
        if date_ics:
            return weighted_huber - ic_weight * torch.stack(date_ics).mean()

    if preds.numel() > 16:
        p_c = preds - preds.mean()
        t_c = targets - targets.mean()
        denom = torch.sqrt(torch.sum(p_c ** 2) * torch.sum(t_c ** 2) + 1e-8)
        corr = torch.sum(p_c * t_c) / denom
        return weighted_huber - ic_weight * corr
    return weighted_huber


def _mean_cross_sectional_ic_np(preds, targets, date_codes):
    """Compute unweighted mean per-date cross-sectional Pearson IC on numpy arrays."""
    if len(preds) < 10:
        return 0.0
    frame = pd.DataFrame({"d": date_codes, "p": preds, "t": targets})
    ics = []
    for _, grp in frame.groupby("d"):
        if len(grp) < 10:
            continue
        p = grp["p"].values
        t = grp["t"].values
        if np.std(p) < 1e-8 or np.std(t) < 1e-8:
            continue
        ics.append(float(np.corrcoef(p, t)[0, 1]))
    return float(np.mean(ics)) if ics else 0.0


def train_attention_net_bag(
    stock_arrays,
    train_meta,
    val_meta,
    input_dim,
    device,
    seeds=(42, 102, 202),
    epochs=28,
    batch_size=256,
    lr=6e-4,
    patience=7,
    horizon=1,
    use_entity_ids=True,
    stride=1,
):
    """Train a multi-seed bagged Global Panel TemporalAttentionNet on RTX 4060 CUDA AMP."""
    use_amp = device.type == "cuda"
    is_long_horizon = horizon >= 5
    train_ds = PanelSequenceDataset(
        stock_arrays, train_meta, LOOKBACK_WINDOW, use_entity_ids=use_entity_ids
    )
    val_ds = PanelSequenceDataset(
        stock_arrays, val_meta, LOOKBACK_WINDOW, use_entity_ids=use_entity_ids
    )

    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, pin_memory=use_amp)
    models = []

    d_model = 64 if is_long_horizon else 128
    gru_hidden = 48 if is_long_horizon else 96
    dropout = 0.40 if is_long_horizon else 0.30
    noise_std = 0.04 if is_long_horizon else 0.02
    opt_lr = 3e-4 if is_long_horizon else lr
    weight_decay = 1e-2 if is_long_horizon else 2e-3
    eff_patience = min(patience, 6) if is_long_horizon else patience
    ic_w = 0.15 if is_long_horizon else 0.05

    for seed in seeds:
        torch.manual_seed(seed)
        if use_amp:
            torch.cuda.manual_seed_all(seed)

        if is_long_horizon:
            train_date_codes = [m[6] for m in train_meta]
            batch_sampler = DateGroupedBatchSampler(
                train_date_codes, dates_per_batch=4, stride=stride, seed=seed
            )
            train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, pin_memory=use_amp)
        else:
            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, pin_memory=use_amp, drop_last=True
            )

        model = TemporalAttentionNet(
            input_dim=input_dim,
            num_stocks=len(SORTED_TICKERS) + 2,
            num_sectors=len(SECTORS) + 2,
            d_model=d_model,
            gru_hidden=gru_hidden,
            num_heads=4,
            dropout=dropout,
            noise_std=noise_std,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        best_val = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(epochs):
            model.train()
            for xb, sb, secb, yb, wb, dcb in train_loader:
                xb = xb.to(device, non_blocking=use_amp)
                sb = sb.to(device, non_blocking=use_amp)
                secb = secb.to(device, non_blocking=use_amp)
                yb = yb.to(device, non_blocking=use_amp)
                wb = wb.to(device, non_blocking=use_amp)
                dcb_dev = dcb.to(device, non_blocking=use_amp) if is_long_horizon else None

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(xb, sb, secb)
                    loss = _huber_ic_loss(preds, yb, wb, date_codes=dcb_dev, ic_weight=ic_w)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            scheduler.step()

            model.eval()
            val_losses = []
            vp_chunks, yt_chunks, dc_chunks = [], [], []
            with torch.no_grad():
                for xb, sb, secb, yb, wb, dcb in val_loader:
                    xb = xb.to(device, non_blocking=use_amp)
                    sb = sb.to(device, non_blocking=use_amp)
                    secb = secb.to(device, non_blocking=use_amp)
                    yb = yb.to(device, non_blocking=use_amp)
                    wb = wb.to(device, non_blocking=use_amp)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        vp = model(xb, sb, secb)
                        vl = _huber_ic_loss(vp, yb, wb, ic_weight=0.0)
                    val_losses.append(float(vl.item()))
                    if is_long_horizon:
                        vp_chunks.append(vp.float().detach().cpu().numpy())
                        yt_chunks.append(yb.float().detach().cpu().numpy())
                        dc_chunks.append(dcb.numpy())

            mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
            if is_long_horizon and vp_chunks:
                val_ic = _mean_cross_sectional_ic_np(
                    np.concatenate(vp_chunks), np.concatenate(yt_chunks), np.concatenate(dc_chunks)
                )
                val_score = mean_val - 0.25 * val_ic
            else:
                val_score = mean_val

            if val_score < best_val - 1e-5:
                best_val = val_score
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= eff_patience:
                    break

        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        models.append(model)

    return models


def predict_attention_net_bag(models, stock_arrays, meta_list, device, use_entity_ids=True):
    """Run bagged inference across meta_list and return (row_indices, predictions)."""
    if not meta_list:
        return [], np.empty((0,), dtype=np.float64)

    use_amp = device.type == "cuda"
    ds = PanelSequenceDataset(
        stock_arrays, meta_list, LOOKBACK_WINDOW, use_entity_ids=use_entity_ids
    )
    loader = DataLoader(ds, batch_size=512, shuffle=False, pin_memory=use_amp)

    bag_preds = []
    for model in models:
        model.eval()
        chunks = []
        with torch.no_grad():
            for xb, sb, secb, _, _, _ in loader:
                xb = xb.to(device, non_blocking=use_amp)
                sb = sb.to(device, non_blocking=use_amp)
                secb = secb.to(device, non_blocking=use_amp)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    out = model(xb, sb, secb)
                chunks.append(out.float().detach().cpu().numpy())
        bag_preds.append(np.concatenate(chunks, axis=0))

    row_indices = [m[5] for m in meta_list]
    return row_indices, np.mean(bag_preds, axis=0).astype(np.float64)


def _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags):
    """Subsample training dates on shifted non-overlapping stride offsets to eliminate serial overlap."""
    if stride <= 1 or tr_date_codes is None or len(tr_date_codes) != len(x_tr):
        return x_tr, y_tr, w_tr
    offset = int((bag_idx * stride) // max(n_bags, 1)) % stride
    mask = (tr_date_codes % stride) == offset
    if int(np.sum(mask)) >= 500:
        return x_tr[mask], y_tr[mask], w_tr[mask]
    return x_tr, y_tr, w_tr


def fit_tabular_family_bag(
    algo,
    x_tr,
    y_tr,
    w_tr,
    x_va,
    y_va,
    use_gpu=False,
    seeds=(42, 102, 202),
    horizon=1,
    tr_date_codes=None,
    stride=1,
    use_entity_ids=True,
):
    """Fit a heavily regularized, multi-offset bagged global panel model family."""
    models = []
    is_long_horizon = horizon >= 5
    n_bags = len(seeds)

    if algo == "LightGBM":
        from lightgbm import LGBMRegressor, early_stopping

        for bag_idx, seed in enumerate(seeds):
            xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
            if is_long_horizon:
                model = LGBMRegressor(
                    objective="huber",
                    alpha=1.5,
                    n_estimators=800,
                    learning_rate=0.01,
                    num_leaves=7 if horizon <= 10 else 4,
                    max_depth=3 if horizon <= 10 else 2,
                    min_child_samples=300 if horizon <= 10 else 400,
                    subsample=0.65,
                    subsample_freq=1,
                    colsample_bytree=0.40 if horizon <= 10 else 0.35,
                    reg_alpha=10.0 if horizon <= 10 else 20.0,
                    reg_lambda=60.0 if horizon <= 10 else 150.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                fit_kwargs = {"callbacks": [early_stopping(stopping_rounds=45, verbose=False)]}
            else:
                model = LGBMRegressor(
                    objective="huber",
                    alpha=1.5,
                    n_estimators=1500,
                    learning_rate=0.015,
                    num_leaves=20,
                    max_depth=5,
                    min_child_samples=150,
                    subsample=0.7,
                    subsample_freq=1,
                    colsample_bytree=0.5,
                    reg_alpha=5.0,
                    reg_lambda=20.0,
                    random_state=seed,
                    n_jobs=-1,
                    verbose=-1,
                )
                fit_kwargs = {"callbacks": [early_stopping(stopping_rounds=60, verbose=False)]}
                if use_entity_ids:
                    fit_kwargs["categorical_feature"] = [x_tr.shape[1] - 2, x_tr.shape[1] - 1]

            model.fit(xb, yb, sample_weight=wb, eval_set=[(x_va, y_va)], **fit_kwargs)
            models.append(model)
        return models

    if algo == "CatBoost":
        from catboost import CatBoostRegressor

        for bag_idx, seed in enumerate(seeds):
            xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
            if is_long_horizon:
                # Robust CPU Huber loss avoids GPU-RMSE outlier blowup on multi-week targets
                m = CatBoostRegressor(
                    loss_function="Huber:delta=1.5",
                    iterations=900,
                    learning_rate=0.015,
                    depth=3 if horizon <= 10 else 2,
                    l2_leaf_reg=60.0 if horizon <= 10 else 150.0,
                    rsm=0.45,
                    random_seed=seed,
                    thread_count=-1,
                    allow_writing_files=False,
                    verbose=False,
                )
                m.fit(xb, yb, sample_weight=wb, eval_set=(x_va, y_va), early_stopping_rounds=50, verbose=False)
                models.append(m)
            else:
                cb_kwargs = dict(
                    iterations=1800,
                    learning_rate=0.02,
                    depth=5,
                    l2_leaf_reg=15.0,
                    random_seed=seed,
                    allow_writing_files=False,
                    verbose=False,
                )
                fitted = None
                if use_gpu:
                    try:
                        m = CatBoostRegressor(loss_function="RMSE", task_type="GPU", **cb_kwargs)
                        m.fit(
                            xb,
                            yb,
                            sample_weight=wb,
                            eval_set=(x_va, y_va),
                            early_stopping_rounds=70,
                            verbose=False,
                        )
                        fitted = m
                    except Exception:
                        fitted = None
                if fitted is None:
                    m = CatBoostRegressor(loss_function="Huber:delta=1.5", thread_count=-1, **cb_kwargs)
                    m.fit(
                        xb,
                        yb,
                        sample_weight=wb,
                        eval_set=(x_va, y_va),
                        early_stopping_rounds=70,
                        verbose=False,
                    )
                    fitted = m
                models.append(fitted)
        return models

    if algo == "XGBoost":
        from xgboost import XGBRegressor

        for bag_idx, seed in enumerate(seeds):
            xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
            if is_long_horizon:
                xgb_kwargs = dict(
                    objective="reg:pseudohubererror",
                    huber_slope=1.5,
                    n_estimators=800,
                    learning_rate=0.01,
                    max_depth=3 if horizon <= 10 else 2,
                    min_child_weight=200 if horizon <= 10 else 300,
                    subsample=0.65,
                    colsample_bytree=0.40 if horizon <= 10 else 0.35,
                    reg_alpha=10.0 if horizon <= 10 else 20.0,
                    reg_lambda=80.0 if horizon <= 10 else 180.0,
                    tree_method="hist",
                    early_stopping_rounds=45,
                    random_state=seed,
                )
            else:
                xgb_kwargs = dict(
                    objective="reg:pseudohubererror",
                    huber_slope=1.5,
                    n_estimators=1500,
                    learning_rate=0.015,
                    max_depth=4,
                    min_child_weight=80,
                    subsample=0.7,
                    colsample_bytree=0.5,
                    reg_alpha=5.0,
                    reg_lambda=20.0,
                    tree_method="hist",
                    early_stopping_rounds=60,
                    random_state=seed,
                )
            fitted = None
            if use_gpu:
                try:
                    m = XGBRegressor(device="cuda", **xgb_kwargs)
                    m.fit(xb, yb, sample_weight=wb, eval_set=[(x_va, y_va)], verbose=False)
                    fitted = m
                except Exception:
                    fitted = None
            if fitted is None:
                m = XGBRegressor(device="cpu", n_jobs=-1, **xgb_kwargs)
                m.fit(xb, yb, sample_weight=wb, eval_set=[(x_va, y_va)], verbose=False)
                fitted = m
            models.append(fitted)
        return models

    if algo == "RandomForest":
        if is_long_horizon:
            for bag_idx, seed in enumerate(seeds):
                xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
                model = RandomForestRegressor(
                    n_estimators=200,
                    max_depth=4 if horizon <= 10 else 3,
                    min_samples_leaf=250,
                    max_features=0.30,
                    max_samples=0.65,
                    random_state=seed,
                    n_jobs=-1,
                )
                model.fit(xb, yb, sample_weight=wb)
                models.append(model)
            return models

        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=80,
            max_features=0.35,
            max_samples=0.70,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(x_tr, y_tr, sample_weight=w_tr)
        return [model]

    if algo == "SVR":
        if is_long_horizon:
            for bag_idx, seed in enumerate(seeds):
                xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
                steps = []
                if use_entity_ids:
                    cont_cols = list(range(x_tr.shape[1] - 2))
                    steps.append(("select_cont", ColumnTransformer([("cont", "passthrough", cont_cols)], remainder="drop")))
                steps.extend(
                    [
                        ("nystroem", Nystroem(kernel="rbf", gamma=0.004, n_components=300, random_state=seed)),
                        ("ridge", RidgeCV(alphas=np.logspace(1, 6, 30))),
                    ]
                )
                model = Pipeline(steps)
                model.fit(xb, yb, ridge__sample_weight=wb)
                models.append(model)
            return models

        cont_cols = list(range(x_tr.shape[1] - 2)) if use_entity_ids else list(range(x_tr.shape[1]))
        model = Pipeline(
            [
                ("select_cont", ColumnTransformer([("cont", "passthrough", cont_cols)], remainder="drop")),
                ("nystroem", Nystroem(kernel="rbf", gamma=0.01, n_components=450, random_state=42)),
                ("ridge", RidgeCV(alphas=np.logspace(0, 5, 25))),
            ]
        )
        model.fit(x_tr, y_tr, ridge__sample_weight=w_tr)
        return [model]

    if algo == "Ridge":
        if is_long_horizon:
            for bag_idx, seed in enumerate(seeds):
                xb, yb, wb = _slice_offset_bag(x_tr, y_tr, w_tr, tr_date_codes, stride, bag_idx, n_bags)
                steps = []
                if use_entity_ids:
                    cont_cols = list(range(x_tr.shape[1] - 2))
                    steps.append(("select_cont", ColumnTransformer([("cont", "passthrough", cont_cols)], remainder="drop")))
                steps.append(("ridge", RidgeCV(alphas=np.logspace(1, 7, 35))))
                model = Pipeline(steps)
                model.fit(xb, yb, ridge__sample_weight=wb)
                models.append(model)
            return models

        cont_cols = list(range(x_tr.shape[1] - 2)) if use_entity_ids else list(range(x_tr.shape[1]))
        model = Pipeline(
            [
                ("select_cont", ColumnTransformer([("cont", "passthrough", cont_cols)], remainder="drop")),
                ("ridge", RidgeCV(alphas=np.logspace(0, 6, 30))),
            ]
        )
        model.fit(x_tr, y_tr, ridge__sample_weight=w_tr)
        return [model]

    raise ValueError(f"Unsupported tabular algorithm: {algo}")


def predict_tabular_bag(models, x_data):
    """Average predictions across all seed models in a bagged family."""
    preds = [np.asarray(m.predict(x_data), dtype=np.float64).reshape(-1) for m in models]
    return np.mean(preds, axis=0)


def fit_quantile_lgbm(x_tr, y_tr, w_tr, x_va, y_va, alpha):
    """Train a global panel LightGBM quantile model (q10 / q50 / q90) on volatility-normalized target z."""
    from lightgbm import LGBMRegressor, early_stopping

    model = LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=24,
        max_depth=5,
        min_child_samples=150,
        subsample=0.75,
        subsample_freq=1,
        colsample_bytree=0.55,
        reg_alpha=3.0,
        reg_lambda=15.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(x_va, y_va)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=False)],
    )
    return model


def _compute_daily_rank_ic(preds, targets, dates):
    """Compute cross-sectional daily Spearman Rank IC and ICIR across valid dates."""
    if dates is None or len(preds) < 10:
        return 0.0, 0.0
    frame = pd.DataFrame({"date": dates, "p": preds, "y": targets})
    frame = frame[np.isfinite(frame["p"]) & np.isfinite(frame["y"])]
    daily_ics = []
    for _, grp in frame.groupby("date"):
        if len(grp) < 10 or grp["p"].nunique() < 3:
            continue
        ic_val, _ = spearmanr(grp["p"].values, grp["y"].values)
        if np.isfinite(ic_val):
            daily_ics.append(float(ic_val))
    if not daily_ics:
        return 0.0, 0.0
    mean_ic = float(np.mean(daily_ics))
    ic_ir = float(mean_ic / (np.std(daily_ics) + 1e-8)) if len(daily_ics) > 5 else 0.0
    return mean_ic, ic_ir


def fit_oof_calibrator(oof_pred, y_true, dates=None, horizon=1):
    """Fit strict zero-centered Bayes-optimal OOF alpha calibrator with RankIC gating for h >= 5.

    Zero-centering prevents a constant positive z-bias from being multiplied by vol_21d,
    while gating on OOF RankIC ensures non-predictive long-horizon models receive gamma = 0.0.
    """
    mask = np.isfinite(oof_pred) & np.isfinite(y_true)
    p = oof_pred[mask]
    y = y_true[mask]
    if len(y) < 100:
        return {"center": 0.0, "bias": 0.0, "gamma": 0.5, "oof_rank_ic": 0.0}

    p_mean = float(np.mean(p))
    y_mean = float(np.mean(y))
    var_p = float(np.var(p))
    cov_py = float(np.mean((p - p_mean) * (y - y_mean)))

    # Ridge-damped optimal slope (automatically shrinks gamma -> 0 if OOF covariance <= 0)
    shrink_den = var_p + 0.02 * max(float(np.var(y)), 1e-4)
    gamma = float(np.clip(cov_py / max(shrink_den, 1e-8), 0.0, 1.25))

    oof_rank_ic = 0.0
    if dates is not None:
        oof_rank_ic, _ = _compute_daily_rank_ic(oof_pred, y_true, dates)
        if horizon >= 5 and oof_rank_ic <= 0.0:
            gamma = 0.0

    return {
        "center": round(p_mean, 5),
        "bias": 0.0,
        "gamma": round(gamma, 5),
        "oof_rank_ic": round(oof_rank_ic, 4),
    }


def fit_shrinkage_stacker(cal_oof_dict, y_true, base_order=STACKING_BASE_ALGORITHMS, horizon=1):
    """Learn diversified zero-intercept non-negative shrinkage meta-weights on alpha OOFs."""
    valid_names = [name for name in base_order if name in cal_oof_dict]
    if not valid_names:
        return {"base_models": [], "weights": [], "intercept": 0.0}

    oof_matrix = np.column_stack([cal_oof_dict[name] for name in valid_names])
    mask = np.all(np.isfinite(oof_matrix), axis=1) & np.isfinite(y_true)
    x_meta = oof_matrix[mask]
    y_meta = y_true[mask]

    n_models = len(valid_names)
    if len(y_meta) < 100:
        w = np.ones(n_models) / n_models
        return {"base_models": valid_names, "weights": w.tolist(), "intercept": 0.0}

    n_active = int(np.sum(np.std(x_meta, axis=0) > 1e-6))
    if horizon >= 5:
        max_w = 0.40 if n_active >= 3 else (0.60 if n_active == 2 else 1.0)
        l2_pen = 5e-3
    else:
        max_w = 1.0
        l2_pen = 1e-3

    # Optimize [w_1, ..., w_k] with 0 <= w_i <= max_w, sum(w) <= 1.0, zero intercept (pure alpha stacker)
    def objective(w):
        pred = x_meta @ w
        mse = np.mean((y_meta - pred) ** 2)
        reg = l2_pen * np.sum(w ** 2)
        return mse + reg

    x0 = np.full(n_models, min(max_w, 0.5 / max(n_models, 1)))
    bounds = [(0.0, max_w)] * n_models
    constraints = {"type": "ineq", "fun": lambda w: 1.0 - np.sum(w)}

    res = minimize(objective, x0=x0, method="SLSQP", bounds=bounds, constraints=constraints)
    if res.success:
        weights = np.clip(res.x, 0.0, max_w)
    else:
        weights = np.full(n_models, min(max_w, 1.0 / n_models))

    return {
        "base_models": valid_names,
        "weights": [round(float(w), 4) for w in weights],
        "intercept": 0.0,
    }


def evaluate_panel_predictions(df_eval, pred_log_ret, horizon=1):
    """Compute institutional quant metrics: DirAcc, MAPE, RMSE, Skill vs Naive, Rank IC, ICIR, L/S Sharpe."""
    actual_ret = df_eval["fwd_ret"].values.astype(np.float64)
    base_price = df_eval["Close"].values.astype(np.float64)
    dates = df_eval["date"].values

    mask = np.isfinite(actual_ret) & np.isfinite(pred_log_ret) & np.isfinite(base_price)
    if np.sum(mask) == 0:
        return {}

    y_t = actual_ret[mask]
    y_p = pred_log_ret[mask]
    p_base = base_price[mask]
    d_valid = dates[mask]

    actual_price = p_base * np.exp(y_t)
    pred_price = p_base * np.exp(y_p)

    price_rmse = float(np.sqrt(np.mean((actual_price - pred_price) ** 2)))
    price_mae = float(np.mean(np.abs(actual_price - pred_price)))
    mape = float(np.mean(np.abs((actual_price - pred_price) / np.maximum(actual_price, 1e-8))) * 100.0)
    ret_rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    naive_ret_rmse = float(np.sqrt(np.mean(y_t ** 2)))
    dir_acc = float(np.mean(np.sign(y_t) == np.sign(y_p)) * 100.0)

    naive_price_rmse = float(np.sqrt(np.mean((actual_price - p_base) ** 2)))
    # Return-space skill (scale-invariant across ₹20 and ₹15,000 stocks!)
    skill_vs_naive = float((1.0 - (ret_rmse / max(naive_ret_rmse, 1e-8))) * 100.0)

    # Cross-sectional daily Rank IC, ICIR, and Top-vs-Bottom Quintile Long-Short Sharpe
    eval_frame = pd.DataFrame({"date": d_valid, "actual": y_t, "pred": y_p})
    daily_ics = []
    ls_returns = []
    for _, grp in eval_frame.groupby("date"):
        if len(grp) < 10 or grp["pred"].nunique() < 3:
            continue
        ic_val, _ = spearmanr(grp["pred"].values, grp["actual"].values)
        if np.isfinite(ic_val):
            daily_ics.append(float(ic_val))

        q_n = max(2, len(grp) // 5)
        ordered = grp.sort_values("pred")
        long_ret = float(ordered["actual"].iloc[-q_n:].mean())
        short_ret = float(ordered["actual"].iloc[:q_n].mean())
        ls_returns.append(long_ret - short_ret)

    if daily_ics:
        rank_ic = float(np.mean(daily_ics))
    elif len(y_t) > 10 and np.std(y_p) > 1e-10:
        ts_ic, _ = spearmanr(y_p, y_t)
        rank_ic = float(ts_ic) if np.isfinite(ts_ic) else 0.0
    else:
        rank_ic = 0.0
    ic_ir = float(rank_ic / (np.std(daily_ics) + 1e-8)) if len(daily_ics) > 5 else 0.0
    if len(ls_returns) > 10 and np.std(ls_returns) > 1e-8:
        ls_sharpe = float((np.mean(ls_returns) / np.std(ls_returns)) * np.sqrt(252.0 / float(horizon)))
    else:
        ls_sharpe = 0.0

    return {
        "directional_accuracy_pct": round(dir_acc, 2),
        "mape_pct": round(mape, 3),
        "price_rmse": round(price_rmse, 2),
        "price_mae": round(price_mae, 2),
        "return_rmse": round(ret_rmse, 5),
        "naive_rmse": round(naive_price_rmse, 2),
        "skill_vs_naive_pct": round(skill_vs_naive, 2),
        "rank_ic": round(rank_ic, 4),
        "ic_ir": round(ic_ir, 3),
        "long_short_sharpe": round(ls_sharpe, 2),
    }


def save_global_family_artifact(models, algo, directory, prefix):
    """Persist multi-seed bagged global family in native format (.pt, .cbm seeds, or .joblib)."""
    if algo == "AttentionNet":
        save_attention_net_bundle(models, os.path.join(directory, f"{prefix}_AttentionNet.pt"))
    elif algo == "CatBoost":
        for idx, m in enumerate(models):
            m.save_model(os.path.join(directory, f"{prefix}_CatBoost_seed{idx}.cbm"))
        # Clean up any leftover seed files from previous runs with larger bag sizes
        for old_idx in range(len(models), 12):
            stale = os.path.join(directory, f"{prefix}_CatBoost_seed{old_idx}.cbm")
            if os.path.exists(stale):
                os.remove(stale)
    else:
        joblib.dump(models, os.path.join(directory, f"{prefix}_{algo}.joblib"))


def update_metrics_summary_batch(horizon_key, global_metrics, per_ticker_metrics):
    """Atomically write GLOBAL_PANEL and per-ticker holdout benchmarks to models/metrics_summary.json."""
    summary = {}
    if os.path.exists(METRICS_SUMMARY_PATH):
        try:
            with open(METRICS_SUMMARY_PATH, "r", encoding="utf-8") as fh:
                summary = json.load(fh)
        except Exception:
            summary = {}

    summary.setdefault("GLOBAL_PANEL", {})[horizon_key] = global_metrics
    for ticker, metrics_map in per_ticker_metrics.items():
        summary.setdefault(ticker, {})[horizon_key] = metrics_map

    tmp_path = f"{METRICS_SUMMARY_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp_path, METRICS_SUMMARY_PATH)


def _train_all_families_parallel(
    x_tab_tr,
    y_tr,
    w_tr,
    x_tab_va,
    y_va,
    stock_arrays,
    tr_seq_meta,
    va_seq_meta,
    seq_input_dim,
    device,
    bag_seeds=(42, 102, 202),
    attn_epochs=26,
    horizon=1,
    tr_date_codes=None,
    stride=1,
    use_entity_ids=True,
):
    """Run CPU families across i7-14650HX threads concurrently while RTX 4060 trains GPU families."""
    use_gpu = device.type == "cuda"
    cpu_algos = ["LightGBM", "RandomForest", "SVR", "Ridge"]
    gpu_algos = ["XGBoost", "CatBoost"]
    trained = {}

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            algo: pool.submit(
                fit_tabular_family_bag,
                algo,
                x_tab_tr,
                y_tr,
                w_tr,
                x_tab_va,
                y_va,
                False,
                bag_seeds,
                horizon,
                tr_date_codes,
                stride,
                use_entity_ids,
            )
            for algo in cpu_algos
        }

        # RTX 4060 GPU workload 1: Multi-seed AttentionNet (FiLM + TCN + BiGRU + Self-Attention)
        trained["AttentionNet"] = train_attention_net_bag(
            stock_arrays,
            tr_seq_meta,
            va_seq_meta,
            input_dim=seq_input_dim,
            device=device,
            seeds=bag_seeds,
            epochs=attn_epochs,
            batch_size=256,
            patience=7,
            horizon=horizon,
            use_entity_ids=use_entity_ids,
            stride=stride,
        )

        # RTX 4060 / full-thread workload 2: Multi-seed XGBoost & CatBoost
        for algo in gpu_algos:
            trained[algo] = fit_tabular_family_bag(
                algo,
                x_tab_tr,
                y_tr,
                w_tr,
                x_tab_va,
                y_va,
                use_gpu=use_gpu,
                seeds=bag_seeds,
                horizon=horizon,
                tr_date_codes=tr_date_codes,
                stride=stride,
                use_entity_ids=use_entity_ids,
            )

        for algo, fut in futures.items():
            trained[algo] = fut.result()

    return trained


def train_global_panel_for_horizon(panel_base, horizon, models_dir, prefix, device, train_quantiles=True):
    """Full end-to-end Global Cross-Sectional Panel pipeline for a single horizon (1, 5, 10, 21, or 63)."""
    horizon_key = "one_day" if horizon == 1 else f"h{horizon}"
    profile = _get_horizon_profile(horizon)
    use_entity_ids = profile["use_entity_ids"]
    stride = profile["stride"]

    print(f"\n=====================================================================")
    print(
        f"[GLOBAL PANEL SOTA] Horizon = {horizon}d ({horizon_key}) | Output prefix: {prefix} | "
        f"entity_ids={use_entity_ids}, stride={stride}d"
    )
    print(f"=====================================================================")

    df_all = attach_horizon_targets(panel_base, horizon)
    valid_df = df_all[df_all["has_target"]].copy()
    valid_dates = np.sort(valid_df["date"].unique())

    embargo = horizon + 1
    test_start_idx = int(len(valid_dates) * 0.80)
    dev_end_idx = max(LOOKBACK_WINDOW + 180, test_start_idx - embargo)

    dev_dates = valid_dates[:dev_end_idx]
    test_dates = valid_dates[test_start_idx:]

    dev_mask = df_all["has_target"] & df_all["date"].isin(dev_dates)
    test_mask = df_all["has_target"] & df_all["date"].isin(test_dates)

    dev_indices = np.where(dev_mask.values)[0]
    test_indices = np.where(test_mask.values)[0]

    print(
        f"Dev rows: {len(dev_indices):,} ({pd.Timestamp(dev_dates[0]).strftime('%Y-%m-%d')} -> "
        f"{pd.Timestamp(dev_dates[-1]).strftime('%Y-%m-%d')}) | "
        f"Unseen Holdout Test rows: {len(test_indices):,} "
        f"({pd.Timestamp(test_dates[0]).strftime('%Y-%m-%d')} -> {pd.Timestamp(test_dates[-1]).strftime('%Y-%m-%d')})"
    )

    x_raw_all = df_all[FEATURE_COLUMNS].values.astype(np.float64)
    entity_cols_all = df_all[["stock_id", "sector_id"]].values.astype(np.float64)
    y_z_all = df_all["target_z"].values.astype(np.float64)
    y_raw_z_all = df_all["raw_target_z"].values.astype(np.float64)
    date_codes_all = pd.factorize(df_all["date"], sort=True)[0]
    weights_all = compute_recency_weights(df_all["date"], half_life_years=profile["half_life_years"])

    # Decoupled return-space market drift (strictly from Dev dates; volatility-independent so it preserves RankIC)
    dev_mkt_series = (
        df_all.iloc[dev_indices][["date", "mkt_fwd_ret"]].drop_duplicates("date")["mkt_fwd_ret"].dropna()
    )
    drift_cap = 0.002 * np.sqrt(float(horizon))
    dev_market_drift_ret = (
        float(np.clip(0.5 * dev_mkt_series.mean(), -drift_cap, drift_cap)) if len(dev_mkt_series) > 0 else 0.0
    )

    # Feature screening strictly inside Dev against market-neutral alpha target (zero test leakage)
    cand_set = set(LONG_HORIZON_CANDIDATE_COLUMNS)
    cand_indices = None if horizon == 1 else [i for i, col in enumerate(FEATURE_COLUMNS) if col in cand_set]
    selected_idx, top_diag = select_stable_features(
        x_raw_all[dev_indices],
        y_z_all[dev_indices],
        max_features=profile["max_features"],
        dates_dev=df_all["date"].values[dev_indices],
        candidate_indices=cand_indices,
        horizon=horizon,
    )
    print(
        f"Selected {len(selected_idx)}/{len(FEATURE_COLUMNS)} highest-stability alpha features | "
        f"Decoupled Dev market drift: {dev_market_drift_ret * 100:+.4f}%"
    )
    if top_diag:
        top_summary = ", ".join(
            f"{FEATURE_COLUMNS[c]}(IC={mic:+.4f},cons={sc:.2f})" for c, mic, _, sc in top_diag[:6]
        )
        print(f"Top Dev stable factors: {top_summary}")

    # --- Step 1: 5-Fold Date-Purged Walk-Forward CV across Global Panel ---
    splits = purged_date_splits(dev_dates, n_splits=5, embargo_days=embargo)
    oof_raw = {algo: np.full(len(df_all), np.nan) for algo in BASE_ALGORITHMS}

    print(f"Running {len(splits)}-fold date-purged walk-forward CV (embargo={embargo}d)...")
    for fold_idx, (tr_dates, va_dates) in enumerate(splits, start=1):
        tr_idx = np.where((df_all["has_target"] & df_all["date"].isin(tr_dates)).values)[0]
        va_idx = np.where((df_all["has_target"] & df_all["date"].isin(va_dates)).values)[0]

        scaler = RobustScaler()
        scaler.fit(x_raw_all[tr_idx])
        scaled_full = np.clip(scaler.transform(x_raw_all), -8.0, 8.0)
        scaled_sel = scaled_full[:, selected_idx]
        x_tab_all = np.hstack([scaled_sel, entity_cols_all]) if use_entity_ids else scaled_sel

        stock_arrays = _build_stock_sequence_store(df_all, scaled_sel)
        tr_seq_meta = _make_sequence_meta(df_all, tr_idx, weights_all, date_codes_all)
        va_seq_meta = _make_sequence_meta(df_all, va_idx, weights_all, date_codes_all)

        fold_models = _train_all_families_parallel(
            x_tab_all[tr_idx],
            y_z_all[tr_idx],
            weights_all[tr_idx],
            x_tab_all[va_idx],
            y_z_all[va_idx],
            stock_arrays,
            tr_seq_meta,
            va_seq_meta,
            seq_input_dim=len(selected_idx),
            device=device,
            bag_seeds=profile["cv_seeds"],
            attn_epochs=profile["cv_attn_epochs"],
            horizon=horizon,
            tr_date_codes=date_codes_all[tr_idx],
            stride=stride,
            use_entity_ids=use_entity_ids,
        )

        for algo in BASE_ALGORITHMS:
            if algo == "AttentionNet":
                r_ids, preds = predict_attention_net_bag(
                    fold_models[algo], stock_arrays, va_seq_meta, device, use_entity_ids=use_entity_ids
                )
                oof_raw[algo][r_ids] = preds
            else:
                oof_raw[algo][va_idx] = predict_tabular_bag(fold_models[algo], x_tab_all[va_idx])

        print(f"  CV Fold {fold_idx}/{len(splits)} complete ({len(tr_idx):,} train / {len(va_idx):,} val rows).")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Fit strict Zero-Centered OOF Alpha Calibrators per model & Shrinkage Stacking Meta-Learner
    calibrators = {}
    oof_calibrated = {}
    dev_dates_arr = df_all["date"].values[dev_indices]
    print("\nLearned Zero-Centered OOF Alpha Calibrators (center, gamma, oof_ic):")
    for algo in BASE_ALGORITHMS:
        cal = fit_oof_calibrator(oof_raw[algo][dev_indices], y_z_all[dev_indices], dates=dev_dates_arr, horizon=horizon)
        calibrators[algo] = cal
        oof_calibrated[algo] = cal["bias"] + cal["gamma"] * (oof_raw[algo][dev_indices] - cal["center"])
        print(
            f"  {algo:<14} -> center={cal['center']:+.5f}, gamma={cal['gamma']:.4f}, "
            f"oof_ic={cal.get('oof_rank_ic', 0.0):+.4f}"
        )

    ensemble_meta = fit_shrinkage_stacker(
        oof_calibrated, y_z_all[dev_indices], base_order=STACKING_BASE_ALGORITHMS, horizon=horizon
    )
    print(
        f"Shrinkage Stacker Weights (sum={sum(ensemble_meta['weights']):.3f}, "
        f"intercept={ensemble_meta['intercept']:+.4f}): "
        f"{dict(zip(ensemble_meta['base_models'], ensemble_meta['weights']))}"
    )

    # --- Step 2: Unbiased Holdout Evaluation on Unseen Chronological 20% Test Period ---
    eval_val_cut = max(LOOKBACK_WINDOW + 120, int(len(dev_dates) * 0.87) - embargo)
    tr_eval_dates = dev_dates[:eval_val_cut]
    va_eval_dates = dev_dates[eval_val_cut + embargo :]

    tr_eval_idx = np.where((df_all["has_target"] & df_all["date"].isin(tr_eval_dates)).values)[0]
    va_eval_idx = np.where((df_all["has_target"] & df_all["date"].isin(va_eval_dates)).values)[0]

    eval_scaler = RobustScaler()
    eval_scaler.fit(x_raw_all[tr_eval_idx])
    scaled_eval_full = np.clip(eval_scaler.transform(x_raw_all), -8.0, 8.0)
    scaled_eval_sel = scaled_eval_full[:, selected_idx]
    x_tab_eval = np.hstack([scaled_eval_sel, entity_cols_all]) if use_entity_ids else scaled_eval_sel

    eval_stock_arrays = _build_stock_sequence_store(df_all, scaled_eval_sel)
    tr_eval_seq = _make_sequence_meta(df_all, tr_eval_idx, weights_all, date_codes_all)
    va_eval_seq = _make_sequence_meta(df_all, va_eval_idx, weights_all, date_codes_all)
    te_eval_seq = _make_sequence_meta(df_all, test_indices, weights_all, date_codes_all)

    eval_models = _train_all_families_parallel(
        x_tab_eval[tr_eval_idx],
        y_z_all[tr_eval_idx],
        weights_all[tr_eval_idx],
        x_tab_eval[va_eval_idx],
        y_z_all[va_eval_idx],
        eval_stock_arrays,
        tr_eval_seq,
        va_eval_seq,
        seq_input_dim=len(selected_idx),
        device=device,
        bag_seeds=profile["prod_seeds"],
        attn_epochs=profile["prod_attn_epochs"] if horizon >= 5 else 26,
        horizon=horizon,
        tr_date_codes=date_codes_all[tr_eval_idx],
        stride=stride,
        use_entity_ids=use_entity_ids,
    )

    df_test = df_all.iloc[test_indices].copy()
    test_vol_scale = df_test["vol_scale"].values.astype(np.float64)
    test_cal_z = {}
    test_pred_ret = {}
    global_metrics = {}

    for algo in BASE_ALGORITHMS:
        if algo == "AttentionNet":
            raw_z_full = np.full(len(df_all), np.nan)
            r_ids, preds = predict_attention_net_bag(
                eval_models[algo], eval_stock_arrays, te_eval_seq, device, use_entity_ids=use_entity_ids
            )
            raw_z_full[r_ids] = preds
            raw_z = raw_z_full[test_indices]
        else:
            raw_z = predict_tabular_bag(eval_models[algo], x_tab_eval[test_indices])

        cal = calibrators[algo]
        cal_z = cal["bias"] + cal["gamma"] * (raw_z - cal["center"])
        test_cal_z[algo] = cal_z
        test_pred_ret[algo] = dev_market_drift_ret + cal_z * test_vol_scale
        global_metrics[algo] = evaluate_panel_predictions(df_test, test_pred_ret[algo], horizon=horizon)

    ens_z_test = np.full(len(test_indices), ensemble_meta["intercept"], dtype=np.float64)
    for base_algo, weight in zip(ensemble_meta["base_models"], ensemble_meta["weights"]):
        ens_z_test += weight * np.nan_to_num(test_cal_z[base_algo], nan=0.0)
    test_pred_ret["Ensemble"] = dev_market_drift_ret + ens_z_test * test_vol_scale
    global_metrics["Ensemble"] = evaluate_panel_predictions(df_test, test_pred_ret["Ensemble"], horizon=horizon)

    print(f"\n>>> UNSEEN 20% HOLDOUT GLOBAL PANEL BENCHMARK ({horizon_key}) <<<")
    print(
        f"  {'Algorithm':<14} {'DirAcc%':>8} {'MAPE%':>7} {'Skill%':>8} "
        f"{'RankIC':>8} {'ICIR':>7} {'L/S Sharpe':>11}"
    )
    for algo in ["Ensemble"] + BASE_ALGORITHMS:
        m = global_metrics.get(algo, {})
        print(
            f"  {algo:<14} {m.get('directional_accuracy_pct', 0):>7.2f}% "
            f"{m.get('mape_pct', 0):>6.2f}% {m.get('skill_vs_naive_pct', 0):>+7.2f}% "
            f"{m.get('rank_ic', 0):>+8.4f} {m.get('ic_ir', 0):>+7.3f} "
            f"{m.get('long_short_sharpe', 0):>+11.2f}"
        )

    # Compute per-ticker holdout metrics for frontend display
    per_ticker_metrics = {}
    for ticker, grp_idx in df_test.groupby("ticker").indices.items():
        sub_df = df_test.iloc[grp_idx]
        per_ticker_metrics[ticker] = {}
        for algo in ["Ensemble"] + BASE_ALGORITHMS:
            per_ticker_metrics[ticker][algo] = evaluate_panel_predictions(
                sub_df, test_pred_ret[algo][grp_idx], horizon=horizon
            )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    # --- Step 3: Production Refit on Full 100% History (latest ~12% used strictly for early stopping) ---
    prod_val_cut = max(LOOKBACK_WINDOW + 180, int(len(valid_dates) * 0.88) - embargo)
    tr_prod_dates = valid_dates[:prod_val_cut]
    va_prod_dates = valid_dates[prod_val_cut + embargo :]

    tr_prod_idx = np.where((df_all["has_target"] & df_all["date"].isin(tr_prod_dates)).values)[0]
    va_prod_idx = np.where((df_all["has_target"] & df_all["date"].isin(va_prod_dates)).values)[0]

    prod_scaler = RobustScaler()
    prod_scaler.fit(x_raw_all[tr_prod_idx])
    scaled_prod_full = np.clip(prod_scaler.transform(x_raw_all), -8.0, 8.0)
    scaled_prod_sel = scaled_prod_full[:, selected_idx]
    x_tab_prod = np.hstack([scaled_prod_sel, entity_cols_all]) if use_entity_ids else scaled_prod_sel

    prod_stock_arrays = _build_stock_sequence_store(df_all, scaled_prod_sel)
    tr_prod_seq = _make_sequence_meta(df_all, tr_prod_idx, weights_all, date_codes_all)
    va_prod_seq = _make_sequence_meta(df_all, va_prod_idx, weights_all, date_codes_all)

    n_prod_seeds = len(profile["prod_seeds"])
    print(f"\nRefitting {n_prod_seeds}-seed bagged production models on {len(tr_prod_idx):,} train + {len(va_prod_idx):,} val rows...")
    prod_models = _train_all_families_parallel(
        x_tab_prod[tr_prod_idx],
        y_z_all[tr_prod_idx],
        weights_all[tr_prod_idx],
        x_tab_prod[va_prod_idx],
        y_z_all[va_prod_idx],
        prod_stock_arrays,
        tr_prod_seq,
        va_prod_seq,
        seq_input_dim=len(selected_idx),
        device=device,
        bag_seeds=profile["prod_seeds"],
        attn_epochs=profile["prod_attn_epochs"],
        horizon=horizon,
        tr_date_codes=date_codes_all[tr_prod_idx],
        stride=stride,
        use_entity_ids=use_entity_ids,
    )

    # Train calibrated P10 / P50 / P90 quantile models concurrently on total return z-score
    q_models = {}
    if train_quantiles:
        with ThreadPoolExecutor(max_workers=3) as q_pool:
            q_futs = {
                name: q_pool.submit(
                    fit_quantile_lgbm,
                    x_tab_prod[tr_prod_idx],
                    y_raw_z_all[tr_prod_idx],
                    weights_all[tr_prod_idx],
                    x_tab_prod[va_prod_idx],
                    y_raw_z_all[va_prod_idx],
                    alpha,
                )
                for name, alpha in [("q10", 0.10), ("q50", 0.50), ("q90", 0.90)]
            }
            q_models = {name: fut.result() for name, fut in q_futs.items()}

    prod_mkt_series = (
        df_all[df_all["has_target"]][["date", "mkt_fwd_ret"]].drop_duplicates("date")["mkt_fwd_ret"].dropna()
    )
    prod_market_drift_ret = (
        float(np.clip(0.5 * prod_mkt_series.mean(), -drift_cap, drift_cap)) if len(prod_mkt_series) > 0 else 0.0
    )

    # Persist all production artifacts
    joblib.dump(prod_scaler, os.path.join(models_dir, f"{prefix}_feature_scaler.joblib"))
    joblib.dump(calibrators, os.path.join(models_dir, f"{prefix}_calibrators.joblib"))
    joblib.dump(ensemble_meta, os.path.join(models_dir, f"{prefix}_Ensemble_meta.joblib"))
    joblib.dump(
        {
            "horizon": horizon,
            "feature_columns": FEATURE_COLUMNS,
            "selected_feature_indices": selected_idx,
            "use_entity_ids": use_entity_ids,
            "horizon_regime": "one_day_sota" if horizon == 1 else "long_horizon_sota",
            "market_drift_ret": round(prod_market_drift_ret, 6),
            "target_mode": "market_neutral_alpha_plus_drift",
            "trained_at": datetime.now().isoformat(),
            "num_stocks": int(df_all["ticker"].nunique()),
            "num_rows": int(len(valid_df)),
        },
        os.path.join(models_dir, f"{prefix}_metadata.joblib"),
    )

    for algo, model_bag in prod_models.items():
        save_global_family_artifact(model_bag, algo, models_dir, prefix)
    for q_name, q_model in q_models.items():
        joblib.dump(q_model, os.path.join(models_dir, f"{prefix}_{q_name}.joblib"))

    update_metrics_summary_batch(horizon_key, global_metrics, per_ticker_metrics)
    print(f"Saved Global Panel production models, calibrators & metrics for {horizon_key}.")

    if device.type == "cuda":
        torch.cuda.empty_cache()
