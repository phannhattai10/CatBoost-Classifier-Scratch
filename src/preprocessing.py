import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    return df


def fit_target_statistics(
    train_df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    smoothing: float = 1.0,
) -> Dict:

    y = train_df[target_col]
    if y.dtype == object or str(y.dtype) == 'str':
        unique_vals = y.dropna().unique()
        pos_label = next(
            (v for v in unique_vals if str(v).strip().lower() in ('yes', '1', 'true')),
            unique_vals[0]
        )
        y = (y == pos_label).astype(float)
    else:
        y = y.astype(float)

    global_mean = float(y.mean())
    mappings: Dict[str, Dict] = {}

    for col in cat_cols:
        col_series = train_df[col].astype(str)
        grp = pd.DataFrame({'cat': col_series, 'target': y}).groupby('cat')
        ts_map = {}
        for category, group in grp:
            n_pos   = group['target'].sum()
            n_total = len(group)
            ts_val  = (n_pos + smoothing * global_mean) / (n_total + smoothing)
            ts_map[category] = float(ts_val)
        mappings[col] = ts_map

    return {
        'global_mean': global_mean,
        'smoothing'  : smoothing,
        'mappings'   : mappings,
    }


def transform_with_target_statistics(
    df: pd.DataFrame,
    ts_params: Dict,
    cat_cols: List[str],
) -> pd.DataFrame:

    df = df.copy()
    global_mean = ts_params['global_mean']
    mappings    = ts_params['mappings']

    for col in cat_cols:
        if col not in mappings:
            raise KeyError(
                f"Cột '{col}' không có trong ts_params. "
                f"Hãy kiểm tra lại cat_cols lúc gọi fit_target_statistics()."
            )
        ts_map = mappings[col]
        df[col] = (df[col]
                   .astype(str)
                   .map(ts_map)
                   .fillna(global_mean)
                   .astype(float))

    return df


def get_feature_lists(
    df: pd.DataFrame,
    target_col: str,
    exclude_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:

    if exclude_cols is None:
        exclude_cols = ['customerID']

    skip = set(exclude_cols) | {target_col}
    num_cols, cat_cols = [], []

    for col in df.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    return num_cols, cat_cols


def fit_missing_value_params(
    train_df: pd.DataFrame,
    num_cols: List[str],
) -> Dict[str, float]:

    fill_params: Dict[str, float] = {}
    for col in num_cols:
        if col in train_df.columns:
            median_val = float(train_df[col].dropna().median())
            fill_params[col] = median_val
    return fill_params


def fill_missing_values(
    df: pd.DataFrame,
    fill_params: Dict[str, float],
) -> pd.DataFrame:

    df = df.copy()
    for col, median_val in fill_params.items():
        if col in df.columns:
            df[col] = df[col].fillna(median_val)
    return df


def fit_preprocessing_params(
    train_df: pd.DataFrame,
    target_col: str = 'Churn',
    exclude_cols: Optional[List[str]] = None,
    smoothing: float = 1.0,
) -> Dict:

    if exclude_cols is None:
        exclude_cols = ['customerID']

    train_clean = clean_total_charges(train_df)

    num_cols, cat_cols = get_feature_lists(
        train_clean, target_col=target_col, exclude_cols=exclude_cols
    )

    # Fit median từ train để fill NaN numerical (tenure, MonthlyCharges, v.v.)
    fill_params = fit_missing_value_params(train_clean, num_cols)

    ts_params = fit_target_statistics(
        train_clean, cat_cols=cat_cols, target_col=target_col, smoothing=smoothing
    )

    return {
        'target_col'  : target_col,
        'exclude_cols': exclude_cols,
        'num_cols'    : num_cols,
        'cat_cols'    : cat_cols,
        'fill_params' : fill_params,
        'ts_params'   : ts_params,
    }


def transform(
    df: pd.DataFrame,
    params: Dict,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:

    target_col  = params['target_col']
    num_cols    = params['num_cols']
    cat_cols    = params['cat_cols']
    fill_params = params.get('fill_params', {})

    # Bước 1: xử lý TotalCharges
    df = clean_total_charges(df)

    # Bước 2: fill NaN numerical bằng median đã fit từ train
    df = fill_missing_values(df, fill_params)

    # Bước 3: KHÔNG còn encode categorical bằng Target Statistics ở đây —
    # cat_cols được giữ nguyên dạng thô (string) để model CatBoost tự xử lý
    # categorical nội bộ. (Xem ghi chú về dtype ngay phía trên hàm này.)
    feature_cols = num_cols + cat_cols
    X = df[feature_cols].copy()
    for col in cat_cols:
        # Ép về object thuần (không phải pandas StringDtype) để tương thích
        # rộng nhất với các thư viện/model bên ngoài; NaN gốc được giữ nguyên.
        X[col] = X[col].where(df[col].isna(), X[col].astype(object).astype(str))

    y = None
    if target_col in df.columns:
        raw_y = df[target_col]
        if raw_y.dtype == object or str(raw_y.dtype) == 'str':
            unique_vals = raw_y.dropna().unique()
            pos_label   = next(
                (v for v in unique_vals if str(v).strip().lower() in ('yes', '1', 'true')),
                unique_vals[0]
            )
            y = (raw_y == pos_label).astype(float).rename(target_col)
        else:
            y = raw_y.astype(float)

    return X, y


def save_params(params: Dict, filepath: str) -> None:

    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_kb = path.stat().st_size / 1024
    print(f'[OK] Da luu params: {path.name}  ({size_kb:.1f} KB)')


def load_params(filepath: str) -> Dict:

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file params: {path.name}\n"
            "Hay chac chan da chay fit_preprocessing_params() va save_params() truoc."
        )
    with open(path, 'rb') as f:
        params = pickle.load(f)
    print(f'[OK] Da tai params: {path.name}')
    print(f'     Cac key: {list(params.keys())}')
    return params

# XỬ LÝ DIRTY DATA (bổ sung sau khi phân tích dataset thật)
# CÁC HẰNG SỐ MAPPING (định nghĩa một lần ở cấp module, dùng chung)

# Mapping cho cột dạng boolean (tự động phát hiện — xem _standardize_categorical_labels)
_BOOLEAN_MAP: Dict[str, str] = {
    '0': 'No',  'false': 'No',  'no': 'No',  'n': 'No',
    '1': 'Yes', 'true':  'Yes', 'yes': 'Yes', 'y': 'Yes',
}
_BOOLEAN_KEYS = frozenset(_BOOLEAN_MAP.keys())

# Mapping tường minh cho các cột có nhãn dài (không rơi vào boolean pattern)
_GENDER_MAP: Dict[str, str] = {
    'm': 'Male', 'male': 'Male', 'f': 'Female', 'female': 'Female',
}

_CONTRACT_MAP: Dict[str, str] = {
    'month-to-month': 'Month-to-month',
    'month to month': 'Month-to-month',
    'one year':       'One year',
    'two year':       'Two year',
    '2 year':         'Two year',
}

_PAYMENT_MAP: Dict[str, str] = {
    'electronic check':          'Electronic check',
    'e-check':                   'Electronic check',
    'mailed check':              'Mailed check',
    'mail check':                'Mailed check',
    'bank transfer (automatic)': 'Bank transfer (automatic)',
    'credit card (automatic)':   'Credit card (automatic)',
    'credit card automatic':     'Credit card (automatic)',
    'credit card':               'Credit card (automatic)',
}

# Cột → dict map tường minh
_EXPLICIT_COL_MAPS: Dict[str, Dict[str, str]] = {
    'gender':        _GENDER_MAP,
    'Contract':      _CONTRACT_MAP,
    'PaymentMethod': _PAYMENT_MAP,
}

# Nhãn nghiệp vụ đặc biệt — KHÔNG được map sang 'No' thông thường.
# "No internet service" khác với "No" (khách hàng không có internet thì
# đương nhiên không có OnlineSecurity — đây là category riêng biệt).
_PRESERVE_LABELS = frozenset({
    'no internet service',
    'no phone service',
})

# Placeholder thiếu dữ liệu → cần convert sang NaN
_MISSING_PLACEHOLDERS = frozenset({
    'unknown', 'nan', 'none', 'null', 'na', 'n/a', '',
})

# Mapping riêng cho target Churn
_TARGET_MAP: Dict[str, str] = {
    'no': 'No',  'n': 'No',  '0': 'No',  'false': 'No',
    'yes': 'Yes', 'y': 'Yes', '1': 'Yes', 'true': 'Yes',
}

# Ngưỡng hợp lý theo nghiệp vụ viễn thông
# max=150 cho MonthlyCharges: phân phối thực tế của dataset Telco Churn gốc đạt
# ~18.25 - 118.75. Chọn 150 làm ngưỡng trần để có biên độ dư (>25% so với max
# thực tế), nhưng vẫn bắt được lỗi nhập liệu như 99999.99 phát hiện khi EDA.
# Ngưỡng này từng bị thiếu (max=None), dẫn đến bỏ sót 1 giá trị lỗi 99999.99
# (customerID '0080-EMYVY') trong dữ liệu thật — đã thêm vào sau khi phát hiện.
_NUMERIC_BOUNDS: Dict[str, Dict[str, Optional[float]]] = {
    'tenure'        : {'min': 0.0, 'max': 100.0},  # đơn vị tháng, tối đa ~8 năm
    'MonthlyCharges': {'min': 0.0, 'max': 150.0},  # max thực tế ~118.75; 150 = biên độ dư
    'TotalCharges'  : {'min': 0.0, 'max': None},    # chỉ reject giá trị âm
}


# ─────────────────────────────────────────────────────────────────────────────
# A. _clean_senior_citizen  (hàm nội bộ, dùng riêng)
# ─────────────────────────────────────────────────────────────────────────────

# Mapping SeniorCitizen về int 0/1
_SENIOR_MAP: Dict[str, int] = {
    '0': 0, 'no': 0, 'false': 0, 'n': 0,
    '1': 1, 'yes': 1, 'true': 1, 'y': 1,
}


def _clean_senior_citizen(df: pd.DataFrame) -> pd.DataFrame:

    if 'SeniorCitizen' not in df.columns:
        return df

    df = df.copy()
    raw = df['SeniorCitizen'].astype(str).str.strip()

    mapped = raw.str.lower().map(_SENIOR_MAP)

    unknown_mask = mapped.isna() & (raw.str.lower() != 'nan') & (raw != '')
    if unknown_mask.any():
        unknown_vals = raw[unknown_mask].unique().tolist()
        print(f"  [CẢNH BÁO] _clean_senior_citizen: {unknown_mask.sum()} dòng "
              f"có giá trị là {unknown_vals} -> đã set thành NaN.")

    df['SeniorCitizen'] = mapped.astype('Int64')  # Int64 hỗ trợ NaN
    return df


def _standardize_categorical_labels(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    for col in df.columns:
        # Chỉ xử lý cột dạng text
        if not (df[col].dtype == object or str(df[col].dtype) in ('string', 'str')):
            continue

        # Strip trước
        s = df[col].astype(str).str.strip()
        # '' (chuỗi rỗng sau strip) → NaN
        s = s.where(s != '', other=np.nan)

        # Chọn map phù hợp cho cột này
        if col in _EXPLICIT_COL_MAPS:
            col_map = _EXPLICIT_COL_MAPS[col]

            def _apply_explicit(val, _cm=col_map):
                if pd.isna(val):
                    return np.nan
                lower = str(val).lower()
                if lower in _MISSING_PLACEHOLDERS:
                    return np.nan
                if lower in _PRESERVE_LABELS:
                    return lower[0].upper() + lower[1:]
                return _cm.get(lower, val)   # giữ nguyên nếu không tìm thấy

            df[col] = s.map(_apply_explicit)

        else:
            # Kiểm tra boolean-like: unique (loại bỏ preserve_labels) ⊆ BOOLEAN_KEYS
            non_null = s.dropna()
            non_null_lower = non_null.str.lower()
            check_set = set(non_null_lower) - _PRESERVE_LABELS - _MISSING_PLACEHOLDERS
            is_boolean_like = bool(check_set) and check_set.issubset(_BOOLEAN_KEYS)

            chosen_map = _BOOLEAN_MAP if is_boolean_like else {}

            def _apply_generic(val, _cm=chosen_map):
                if pd.isna(val):
                    return np.nan
                lower = str(val).lower()
                if lower in _MISSING_PLACEHOLDERS:
                    return np.nan
                if lower in _PRESERVE_LABELS:
                    return lower[0].upper() + lower[1:]
                if _cm:
                    return _cm.get(lower, val)
                return val   # cột không boolean-like, không explicit → giữ nguyên

            df[col] = s.map(_apply_generic)

    return df


def clean_target_column(
    df: pd.DataFrame,
    target_col: str = 'Churn',
) -> pd.DataFrame:

    if target_col not in df.columns:
        raise KeyError(f"Cột target '{target_col}' không tồn tại trong DataFrame.")

    df = df.copy()
    raw = df[target_col].astype(str).str.strip()

    print(f"[clean_target_column] Cột '{target_col}' — giá trị gốc:")
    for val, cnt in raw.value_counts(dropna=False).items():
        print(f"    {repr(val):20s}: {cnt:6,}")

    # Map về chuẩn
    mapped = raw.str.lower().map(_TARGET_MAP)

    # Phát hiện giá trị không map được (không phải NaN gốc)
    unknown_mask = mapped.isna() & raw.notna() & (raw.str.strip() != '') & (raw != 'nan')
    if unknown_mask.any():
        unknown_vals = raw[unknown_mask].unique().tolist()
        print(f"\n  [CẢNH BÁO] Còn {unknown_mask.sum()} dòng có giá trị target là:\n"
              f"  {unknown_vals}\n"
              f"  -> Những dòng này sẽ bị set thành NaN. Kiểm tra lại dữ liệu!")
    else:
        print(f"\n  -> Map thành công: chỉ còn 'Yes' / 'No'.")

    df[target_col] = mapped

    print(f"\n  Giá trị sau khi map:")
    for val, cnt in df[target_col].value_counts(dropna=False).items():
        pct = cnt / len(df) * 100
        print(f"    {repr(val):12s}: {cnt:6,}  ({pct:.2f}%)")

    return df


def _invalidate_impossible_numeric_values(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()
    total = 0

    print("[_invalidate_impossible_numeric_values] Kết quả:")
    for col, bounds in _NUMERIC_BOUNDS.items():
        if col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        before = int(df[col].isna().sum())
        lo, hi = bounds['min'], bounds['max']

        if lo is not None:
            df.loc[df[col] < lo, col] = np.nan
        if hi is not None:
            df.loc[df[col] > hi, col] = np.nan

        n_inv = int(df[col].isna().sum()) - before
        total += n_inv

        lo_str = f">= {lo}" if lo is not None else "không có chặn dưới"
        hi_str = f"<= {hi}" if hi is not None else "không có chặn trên"
        valid_range = df[col].dropna()
        stat_str = (f"min={valid_range.min():.2f}, max={valid_range.max():.2f}"
                    if len(valid_range) > 0 else "không có giá trị hợp lệ")
        print(f"  {col:20s}: {n_inv:4d} dòng bị invalidate  "
              f"(hợp lệ: {lo_str}, {hi_str}) | sau: {stat_str}")

    print(f"  Tổng: {total} giá trị đã được set thành NaN.")
    return df


def clean_raw_data(
    df: pd.DataFrame,
    drop_customer_id: bool = True,
    target_col: str = 'Churn',
) -> pd.DataFrame:

    df = df.copy()

    # ── Bước 0: Dedup — log và xóa dòng trùng lặp hoàn toàn ─────────────────
    if 'customerID' in df.columns:
        n_id_dup = int(df['customerID'].duplicated().sum())
        if n_id_dup > 0:
            dup_ids = df[df['customerID'].duplicated(keep=False)]['customerID'].unique()
            print(f"[clean_raw_data] Bước 0: Phát hiện {n_id_dup} customerID bị trùng:")
            id_counts = df[df['customerID'].isin(dup_ids)].groupby('customerID').size()
            for cid, cnt in id_counts.items():
                print(f"  {cid}: {cnt} dòng")
        else:
            print("[clean_raw_data] Bước 0: Không có customerID trùng.")

    n_before = len(df)
    df = df.drop_duplicates(keep='first')
    n_full_dup = n_before - len(df)
    if n_full_dup > 0:
        print(f"[clean_raw_data] Bước 0: Đã xóa {n_full_dup} dòng trùng lặp hoàn toàn.")
    else:
        print("[clean_raw_data] Bước 0: Không có dòng trùng lặp hoàn toàn.")

    # ── Bước 1: Drop customerID ───────────────────────────────────────────────
    if drop_customer_id and 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        print("[clean_raw_data] Bước 1: Drop cột 'customerID'.")
    else:
        print("[clean_raw_data] Bước 1: Giữ nguyên customerID.")

    # ── Bước 2: Strip khoảng trắng mọi cột object ────────────────────────────
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
    print(f"[clean_raw_data] Bước 2: Strip khoảng trắng {len(obj_cols)} cột object.")

    # ── Bước 3: Convert placeholder thiếu dữ liệu → NaN ──────────────────────
    n_placeholder = 0
    for col in obj_cols:
        if col not in df.columns:
            continue
        mask = df[col].str.lower().isin(_MISSING_PLACEHOLDERS)
        n = int(mask.sum())
        if n > 0:
            df.loc[mask, col] = np.nan
            n_placeholder += n
            print(f"  {col}: {n} placeholder -> NaN")
    print(f"[clean_raw_data] Bước 3: {n_placeholder} placeholder đã convert sang NaN.")

    # ── Bước 4: Chuẩn hoá SeniorCitizen về int {0,1} TRƯỚC standardize ──────
    df = _clean_senior_citizen(df)
    print("[clean_raw_data] Bước 4: _clean_senior_citizen() xong.")

    # ── Bước 5: Chuẩn hoá nhãn categorical ───────────────────────────────────
    df = _standardize_categorical_labels(df)
    print("[clean_raw_data] Bước 5: Chuẩn hoá nhãn categorical xong.")

    # ── Bước 6: Tự động convert các cột dạng số ──────────────────────────────
    obj_cols_now = [c for c in df.select_dtypes(include=['object']).columns
                    if c != target_col]
    converted = []
    for col in obj_cols_now:
        try:
            df[col] = pd.to_numeric(df[col], errors='raise')
            converted.append(col)
        except (ValueError, TypeError):
            pass
    msg = f"{converted}" if converted else "không có cột mới"
    print(f"[clean_raw_data] Bước 6: Convert sang numeric: {msg}")

    # ── Bước 7: Invalidate giá trị số phi lý ─────────────────────────────────
    df = _invalidate_impossible_numeric_values(df)
    print("[clean_raw_data] Bước 7: Invalidate giá trị phi lý xong.")

    # ── Bước 8: Xử lý TotalCharges ───────────────────────────────────────────
    if 'TotalCharges' in df.columns:
        df = clean_total_charges(df)
        print("[clean_raw_data] Bước 8: clean_total_charges() xong.")

    return df


def resolve_duplicate_customer_ids(
    df: pd.DataFrame,
    id_col: str = 'customerID',
    check_cols: Optional[List[str]] = None,
) -> pd.DataFrame:

    if check_cols is None:
        check_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    if id_col not in df.columns:
        raise KeyError(
            f"Cột '{id_col}' không tồn tại trong DataFrame. Hãy chắc chắn đã "
            f"gọi clean_raw_data(df, drop_customer_id=False) trước khi gọi "
            f"resolve_duplicate_customer_ids()."
        )

    df = df.copy()
    n_before = len(df)

    n_dup_ids = int(df[id_col].duplicated(keep=False).sum())
    if n_dup_ids == 0:
        print(f"[resolve_duplicate_customer_ids] Không còn {id_col} nào bị "
              f"trùng (khác dữ liệu) sau clean_raw_data.")
        return df

    df['_n_nan'] = df[check_cols].isna().sum(axis=1)
    df = df.sort_values('_n_nan', kind='stable')
    df = df.drop_duplicates(subset=[id_col], keep='first')
    df = df.drop(columns=['_n_nan'])

    n_after = len(df)
    n_removed = n_before - n_after
    print(f"[resolve_duplicate_customer_ids] Đã loại {n_removed} dòng trùng "
          f"{id_col} (giữ bản ghi ít NaN nhất trong {check_cols}; "
          f"hoà -> giữ bản ghi xuất hiện trước trong df).")

    assert df.duplicated(subset=[id_col]).sum() == 0, (
        f"FAIL: vẫn còn {id_col} bị trùng sau resolve_duplicate_customer_ids()."
    )

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    # 1. tenure_group
    bins   = [-0.1, 6, 12, 24, 36, 48, 60, 100]
    labels = ['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61+']
    df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels).astype(str)
    df.loc[df['tenure'].isna(), 'tenure_group'] = np.nan

    # 2. num_addon_services
    addon_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                  'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['num_addon_services'] = (df[addon_cols] == 'Yes').sum(axis=1)

    # 3. avg_monthly_spend (fallback = MonthlyCharges khi tenure = 0)
    safe_tenure = df['tenure'].replace(0, np.nan)
    avg_spend   = df['TotalCharges'] / safe_tenure
    df['avg_monthly_spend'] = avg_spend.fillna(df['MonthlyCharges'])

    return df

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    RAW = 'data/raw/Telco_Customer_Churn.csv'
    TRAIN = 'data/processed/train_raw.csv'
    TEST  = 'data/processed/test_raw.csv'

    print('=== Tự kiểm thử preprocessing.py ===')
    print()

    # -- Test clean_target_column --
    df_raw = pd.read_csv(RAW)
    df_raw = clean_target_column(df_raw, 'Churn')
    assert set(df_raw['Churn'].dropna().unique()) <= {'Yes', 'No'}, 'FAIL: target'
    print('PASS: clean_target_column')

    # -- Test clean_raw_data --
    df_clean = clean_raw_data(df_raw.copy())
    assert df_clean['SeniorCitizen'].dtype in (np.int64, np.float64, 'Int64'), \
        f"FAIL: SeniorCitizen dtype = {df_clean['SeniorCitizen'].dtype}"
    assert set(df_clean['gender'].dropna().unique()) <= {'Male', 'Female'}
    assert set(df_clean['Contract'].dropna().unique()) <= {
        'Month-to-month', 'One year', 'Two year'}
    tc_nan = df_clean['TotalCharges'].isna().sum()
    assert tc_nan == 0, f'FAIL: TotalCharges NaN = {tc_nan}'
    ten_max = df_clean['tenure'].dropna().max()
    assert ten_max <= 100, f'FAIL: tenure max = {ten_max}'
    print('PASS: clean_raw_data')

    # -- Test fit + transform: assert zero NaN trong X --
    if not pd.io.common.file_exists(TRAIN):
        print('SKIP: train_raw.csv không tồn tại, bỏ qua test fit/transform.')
        sys.exit(0)

    train_df = pd.read_csv(TRAIN)
    test_df  = pd.read_csv(TEST)

    params = fit_preprocessing_params(train_df, target_col='Churn')
    print(f'fill_params: {params["fill_params"]}')

    X_train, y_train = transform(train_df, params)
    X_test,  y_test  = transform(test_df,  params)

    # Chỉ assert 0 NaN trên num_cols -- cat_cols giờ giữ dạng thô (string),
    # KHÔNG còn được TS-encode trong transform() nên có thể còn NaN (vd.
    # 'tenure_group' khi tenure gốc là NaN). Xem ghi chú trong hàm transform().
    num_cols = params['num_cols']
    nan_train_num = int(X_train[num_cols].isna().sum().sum())
    nan_test_num  = int(X_test[num_cols].isna().sum().sum())
    assert nan_train_num == 0, f'FAIL: X_train[num_cols] có {nan_train_num} NaN'
    assert nan_test_num  == 0, f'FAIL: X_test[num_cols] có {nan_test_num} NaN'
    print(f'PASS: transform — X_train[num_cols] {X_train[num_cols].shape}, '
          f'NaN={nan_train_num}; X_test[num_cols] {X_test[num_cols].shape}, '
          f'NaN={nan_test_num}')

    cat_cols = params['cat_cols']
    nan_train_cat = int(X_train[cat_cols].isna().sum().sum())
    nan_test_cat  = int(X_test[cat_cols].isna().sum().sum())
    print(f'INFO: X_train[cat_cols] NaN={nan_train_cat} (có thể > 0, không '
          f'phải lỗi); X_test[cat_cols] NaN={nan_test_cat}')
    print()
    print('=== TẤT CẢ TỰ KIỂM THỬ ĐÃ PASS ===')
