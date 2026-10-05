# Báo cáo Tổng quan Pipeline Dữ liệu (Data Pipeline Summary)

## Cấu trúc thư mục và vai trò của từng file

Dưới đây là cấu trúc toàn bộ dự án và mô tả vai trò của từng file/thư mục:

```text
project/
├── data/
│   ├── raw/
│   │   └── Telco_Customer_Churn.csv       # File dữ liệu thô ban đầu (7,143 dòng x 21 cột)
│   └── processed/
│       ├── train_raw.csv                  # Tập Train sau làm sạch & split (5,638 dòng, 80%)
│       ├── test_raw.csv                   # Tập Test sau làm sạch & split (1,410 dòng, 20%)
│       └── preprocessing_params.pkl       # Tham số tiền xử lý fit trên Train (median, TS mapping)
├── notebooks/
│   ├── 01_split.ipynb                     # Notebook làm sạch dữ liệu và chia tách Stratified Train/Test (80/20)
│   ├── 02_eda.ipynb                       # Notebook phân tích khám phá dữ liệu (EDA) trên tập Train (5,638 dòng)
│   ├── 03_feature_engineering.ipynb       # Notebook tạo feature mới và fit tham số tiền xử lý trên Train
│   ├── 04_train_model.ipynb               # Notebook tách Val từ Train (80/20), huấn luyện CatBoost từ scratch
│   └── 05_compare_with_catboost_lib.ipynb # Notebook tách Val từ Train, so sánh đối chuẩn với thư viện CatBoost chính thức
├── src/
│   ├── preprocessing.py                   # Module chứa toàn bộ logic làm sạch, kỹ thuật đặc trưng & biến đổi
│   ├── Model.py                           # Module cài đặt mô hình CatBoost & Oblivious Trees từ scratch
│   ├── TrainModel.py                      # Script huấn luyện mô hình và đánh giá validation
│   ├── FinalTest.py                       # Script đánh giá độc lập mô hình trên test_raw.csv
│   └── Visualize.py                       # Script trực quan hóa hiệu năng mô hình
├── reports/
│   ├── data_pipeline_summary.md           # Báo cáo tóm tắt pipeline dữ liệu và số liệu thực tế
│   └── figures/                           # Thư mục chứa 10 biểu đồ trích xuất từ EDA và Feature Engineering
└── requirements.txt                       # Danh sách các thư viện phụ thuộc của dự án
```

---

## Ràng buộc của đề bài & Giải pháp nhóm đề ra

### Ràng buộc buổi đầu thầy yêu cầu
- **Không sử dụng thư viện scikit-learn** cho phần cài đặt thuật toán chính.
- **Tập test thực tế luôn là tập thô** — không được tiền xử lý hay tham gia vào quá trình fit thông số.
- **Sử dụng các kỹ thuật chống rò rỉ dữ liệu (data leakage prevention)** trong toàn bộ pipeline.

### Giải pháp của nhóm đề ra

| Ràng buộc | Giải pháp |
| :--- | :--- |
| Không dùng scikit-learn | Chỉ sử dụng `scikit-learn` để **so sánh kết quả** . Toàn bộ thuật toán, pipeline tiền xử lý được cài đặt thuần NumPy/Pandas. |
| Tập test thực tế là tập thô | **Split tập thô đầu tiên** (Notebook 01) để chia 2 tập train/test ở dạng thô. Tập test được giữ độc lập hoàn toàn. Nếu cần tập val ở Notebook 04/05 thì split val từ train. |
| Chống rò rỉ dữ liệu | **Chỉ fit tham số duy nhất trên tập train**, xuất ra file `.pkl` để có thể tái sử dụng biến đổi dữ liệu mới trong tương lai một cách nhất quán. Val và test tuyệt đối không tham gia |

---

## 1. Tổng quan luồng xử lý dữ liệu

Pipeline xử lý dữ liệu được thiết kế thành chuỗi 3 giai đoạn nối tiếp nhau, tách bạch rõ ràng trách nhiệm để tránh rò rỉ dữ liệu (Data Leakage):

```text
      ┌───────────────────────┐
      │    01_split.ipynb     │  Input: Telco_Customer_Churn.csv (Raw)
      │ (Làm sạch & Split 8/2)│  Output: train_raw.csv (80%), test_raw.csv (20%)
      └───────────────────────┘
                  │
                  ├──────────────────────────────────────┐
                  ▼ (Chỉ dùng train_raw.csv)             ▼ (Không mở ở EDA & FE)
      ┌───────────────────────┐              ┌────────────────────────┐
      │     02_eda.ipynb      │              │ test_raw.csv           │
      │  (Phân tích EDA)      │              │ (Giữ độc lập hoàn toàn)│
      └───────────────────────┘              └────────────────────────┘
                  │                                      │
                  ▼                                      │
      ┌───────────────────────┐                          │
      │ 03_feature_eng.ipynb  │  Input: train_raw.csv    │
      │(Feature Eng & Fit TS) │  Output: params.pkl, X/y │
      └───────────────────────┘                          │
                  │                                      │
                  ▼                                      │
      ┌───────────────────────┐                          │
      │  04_train_model.ipynb │                          │
      │ (Tách val từ train,   │ ◄────────────────────────┘
      │  train & evaluate)    │   Đánh giá cuối cùng trên test_raw
      └───────────────────────┘
```

- **Bước 1 — `01_split.ipynb` (Làm sạch & Chia tách tập dữ liệu):**
  - *Input:* File CSV gốc `data/raw/Telco_Customer_Churn.csv` (7,143 dòng x 21 cột).
  - *Xử lý:* Chuẩn hóa nhãn target `Churn` và các biến phân loại, invalidate giá trị số phi lý (tenure > 100, MonthlyCharges > 150), xóa duplicate hoàn toàn và xử lý trùng lặp `customerID` khác dữ liệu. Thực hiện **Stratified Split (80 / 20)** thuần NumPy.
  - *Output:* 2 file CSV đã làm sạch nhưng chưa encode feature: `train_raw.csv` (80%), `test_raw.csv` (20%) trong `data/processed/`. Tập `val` (nếu cần) sẽ được tách từ `train` ở Notebook 04/05.
- **Bước 2 — `02_eda.ipynb` (Phân tích khám phá dữ liệu):**
  - *Input:* Chỉ đọc duy nhất `data/processed/train_raw.csv` (5,638 dòng x 20 cột).
  - *Xử lý:* Phân tích phân phối đơn biến/đa biến, kiểm tra missing values, phân tích ngoại lai (IQR), ma trận tương quan Pearson, churn rate theo nhóm hợp đồng, tenure, dịch vụ gia tăng.
  - *Output:* 10 hình ảnh trực quan hóa lưu tại `reports/figures/` và bộ nhận định (insights) làm cơ sở thiết kế feature engineering.
- **Bước 3 — `03_feature_engineering.ipynb` (Kỹ thuật đặc trưng & Fit tiền xử lý):**
  - *Input:* Chỉ đọc `train_raw.csv` (5,638 dòng) — tuyệt đối không đọc `test_raw.csv`.
  - *Xử lý:* Tạo 3 feature mới (`tenure_group`, `num_addon_services`, `avg_monthly_spend`) bằng hàm thuần `engineer_features()`. Fit tham số median và Target Statistics **chỉ trên tập train**. Xuất ma trận đặc trưng $X_{\text{train}}$ và nhãn $y_{\text{train}} \in \{0.0, 1.0\}$.
  - *Output:* File tham số `data/processed/preprocessing_params.pkl` và `scale_pos_weight = 2.7687`.

---

## 2. Làm sạch dữ liệu (từ `01_split.ipynb`)

Các số liệu thực tế được ghi nhận chính xác từ quá trình thực thi các hàm trong preprocessing.py:

### 2.1. Chuẩn hóa cột nhãn Target `Churn` (`clean_target_column`)
Dữ liệu thô ban đầu có 7,143 dòng với **9 biến thể khác nhau** của giá trị Churn (do khoảng trắng thừa, chữ hoa/thường, nhãn số hoặc boolean):

| Giá trị gốc trong dữ liệu thô | Số lượng dòng ghi nhận | Giá trị chuẩn hóa sau khi Map |
|---|---:|---|
| `'No'` | 5,235 | `'No'` |
| `'Yes'` | 1,900 | `'Yes'` |
| `' No'` (khoảng trắng phía trước) | 2 | `'No'` |
| `'1'` | 1 | `'Yes'` |
| `'False'` | 1 | `'No'` |
| `'NO'` | 1 | `'No'` |
| `'YES'` | 1 | `'Yes'` |
| `'no'` | 1 | `'No'` |
| `'N'` | 1 | `'No'` |
| **Tổng cộng** | **7,143** | **'No': 5,241 (73.37%), 'Yes': 1,902 (26.63%)** |

### 2.2. Xử lý bản ghi trùng lặp và xung đột định danh (`clean_raw_data` & `resolve_duplicate_customer_ids`)
- **Dòng trùng lặp hoàn toàn (Full duplicate):**
  - Ở bước đầu của `clean_raw_data()`, xóa **18 dòng** trùng lặp hoàn toàn (giảm từ 7,143 xuống 7,125 dòng).
  - Sau khi chuẩn hóa nhãn và kiểu dữ liệu, kiểm tra phát hiện thêm **47 dòng** trùng lặp hoàn toàn và đã loại bỏ (giảm từ 7,125 xuống 7,078 dòng).
- **Xử lý `customerID` bị trùng nhưng dữ liệu khác nhau:**
  - Phát hiện **77 customerID** xuất hiện nhiều hơn 1 lần (chiếm 92 lượt bản ghi).
  - Hàm `resolve_duplicate_customer_ids()` đã loại bỏ **30 dòng** dư thừa (giảm từ 7,078 xuống 7,048 dòng).
  - *Tiêu chí chọn bản ghi giữ lại:* Đếm số lượng giá trị thiếu (`NaN`) trên 3 cột số quan trọng `['tenure', 'MonthlyCharges', 'TotalCharges']` và **ưu tiên giữ bản ghi có ít giá trị NaN nhất**; nếu số lượng NaN bằng nhau thì giữ bản ghi xuất hiện trước (ổn định thứ tự). Sau khi xử lý xong duplicate theo ID, cột `customerID` được loại bỏ hoàn toàn.

### 2.3. Vô hiệu hóa các giá trị số phi lý (`_invalidate_impossible_numeric_values`)
Hệ thống thiết lập chặn ngưỡng hợp lý theo nghiệp vụ viễn thông:
- `tenure`: Ngưỡng hợp lệ $[0.0, 100.0]$ tháng (tương đương tối đa ~8 năm). Có **6 dòng** vượt ngưỡng bị chuyển thành `NaN` (sau khi xử lý: $\min = 0.00, \max = 72.00$).
- `MonthlyCharges`: Ngưỡng hợp lệ $[0.0, 150.0]$ USD (phân phối chuẩn viễn thông $\approx 18.25 - 118.75$, ngưỡng 150 để chặn các lỗi nhập liệu như 99999.99 phát hiện trong dữ liệu). Có **8 dòng** vượt ngưỡng bị chuyển thành `NaN` (sau khi xử lý: $\min = 18.25, \max = 118.75$).
- **Tổng cộng:** **14 giá trị số phi lý** đã được chuyển thành `NaN` an toàn để impute ở bước sau.

### 2.4. Xử lý cột `TotalCharges` (`clean_total_charges`)
- Ban đầu `TotalCharges` là kiểu `object` do có **11 dòng** chứa placeholder khoảng trắng rỗng (`" "`).
- Ở Bước 3 của `clean_raw_data()`, 11 dòng này được ép sang `NaN`, kèm theo 5 placeholder ở các cột khác (`OnlineBackup`: 2, `DeviceProtection`: 1, `TechSupport`: 1, `StreamingTV`: 1) nâng tổng số placeholder xử lý lên **16 dòng**.
- `clean_total_charges()` thực hiện `pd.to_numeric(errors='coerce')` và `fillna(0.0)` (hầu hết là khách hàng mới có `tenure = 0` nên tổng tiền bằng 0). Kết quả: **0 dòng NaN** trong `TotalCharges` sau bước làm sạch.

---

## 3. Kết quả Train / Test Split (80 / 20)

Sau khi hoàn tất làm sạch và khử trùng lặp (tập FULL còn lại đúng **7,048 dòng**), dữ liệu được chia tách bằng hàm tự xây dựng `stratified_split()` với tỷ lệ **80% / 20%**, cố định `random_state = 42`.

### Bảng phân phối tập dữ liệu và tỷ lệ nhãn Churn

| Tập dữ liệu | Số lượng dòng | Tỷ lệ tập (%) | Churn = No (dòng) | Churn = No (%) | Churn = Yes (dòng) | Churn = Yes (%) | File lưu trữ |
|---|---:|---:|---:|---:|---:|---:|---|
| **FULL (gốc)** | **7,048** | 100.0% | 5,178 | 73.47% | 1,870 | 26.53% | `data/raw/` |
| **TRAIN** | **5,638** | 80.0% | 4,142 | 73.47% | 1,496 | 26.53% | `data/processed/train_raw.csv` (713.8 KB) |
| **TEST** | **1,410** | 20.0% | 1,036 | 73.48% | 374 | 26.52% | `data/processed/test_raw.csv` (181.2 KB) |

> **Xác nhận Stratified Split:** Tỷ lệ nhãn `Churn=Yes` trong 2 tập Train (26.53%) và Test (26.52%) xấp xỉ tuyệt đối với tỷ lệ của tập FULL (26.53%), đảm bảo tính đại diện đồng đều và không làm lệch phân phối thực nghiệm. Nếu cần tập validation (ở Notebook 04 và 05), tập `val` sẽ được tách trực tiếp từ `train_raw.csv`.

---

## 4. Insight chính từ EDA (`02_eda.ipynb`)

Phân tích khám phá được thực hiện nghiêm ngặt **chỉ trên tập Train (5,638 dòng)**. Các phát hiện định lượng quan trọng dẫn tới quyết định thiết kế đặc trưng bao gồm:

1. **Giai đoạn khách hàng mới có tỷ lệ Churn đột biến (Tenure vs Churn):**
   - Tương quan Pearson giữa `tenure` và `Churn` mang giá trị âm rõ rệt: $r = -0.3478$.
   - Phân tích theo nhóm thời gian cho thấy nhóm khách hàng có `tenure` từ **0 - 6 tháng** có tỷ lệ rời bỏ lên tới **47.3%** (cao vượt trội so với mức trung bình 26.54%), trong khi nhóm `tenure > 60 tháng` chỉ có churn rate khoảng **6.8%**.
   - *Quyết định FE:* Tạo đặc trưng `tenure_group` để bin hóa thời gian sử dụng, giúp mô hình bắt được các mốc chuyển tiếp hành vi phi tuyến tính.
2. **Ảnh hưởng bảo vệ của hệ sinh thái dịch vụ gia tăng (Add-on Services):**
   - Khách hàng không kích hoạt các dịch vụ giá trị gia tăng như `OnlineSecurity` và `TechSupport` có tỷ lệ rời đi lần lượt là **41.59%** và **41.2%**, cao gấp gần 3 lần so với nhóm có đăng ký (`OnlineSecurity=Yes`: 15.07%, `TechSupport=Yes`: 15.1%).
   - *Quyết định FE:* Tạo đặc trưng `num_addon_services` đếm tổng số dịch vụ hỗ trợ/bảo mật mà khách hàng kích hoạt (thang điểm từ 0 đến 6).
3. **Hiện tượng đa cộng tuyến giữa `tenure` và `TotalCharges`:**
   - Tương quan giữa `tenure` và `TotalCharges` rất cao ($r \approx 0.83$), do tổng chi phí tích lũy tỉ lệ thuận tự nhiên với số tháng sử dụng.
   - *Quyết định FE:* Tạo đặc trưng `avg_monthly_spend` $= \text{TotalCharges} / \text{tenure}$ (với fallback bằng `MonthlyCharges` khi `tenure = 0`) nhằm đo lường mức chi tiêu thực tế bình quân hàng tháng, triệt tiêu sự phụ thuộc tuyến tính vào thời gian hợp đồng.
4. **Mức độ mất cân bằng lớp (Class Imbalance):**
   - Tỷ lệ `No` : `Yes` trong tập Train là $4,142 : 1,496 \approx \mathbf{2.769 : 1}$ (mất cân bằng mức độ vừa).
   - *Quyết định FE & Model:* Đề xuất sử dụng trọng số lớp $\text{scale\_pos\_weight} = 2.7687$ cho hàm mất mát và ưu tiên đánh giá bằng metric AUC-ROC và F1-score thay vì Accuracy.
5. **Chất lượng dữ liệu và ngoại lai:**
   - Kiểm tra ngoại lai (IQR) trên các cột số: `tenure` (0 outlier, phạm vi $[0, 72]$), `MonthlyCharges` (0 outlier, phạm vi $[18.25, 118.65]$), `TotalCharges` (0 outlier). `SeniorCitizen` là biến nhị phân $\{0, 1\}$.

---

## 5. Feature Engineering & Phân loại cột (`03_feature_engineering.ipynb`)

### 5.1. Ba đặc trưng mới được tạo ra (`engineer_features`)

Hàm `engineer_features()` là hàm biến đổi thuần túy theo từng dòng (row-wise), áp dụng đồng nhất cho mọi tập:

| Tên Feature mới | Kiểu dữ liệu | Công thức / Logic tính toán | Mục đích nghiệp vụ |
|---|---|---|---|
| `tenure_group` | Categorical (`str`) | `pd.cut(df['tenure'], bins=[-0.1, 6, 12, 24, 36, 48, 60, 100], labels=['0-6', '7-12', '13-24', '25-36', '37-48', '49-60', '61+'])` (giữ `NaN` nếu `tenure` thiếu) | Bắt trọn bước nhảy tỷ lệ Churn theo chu kỳ tuổi đời khách hàng (đặc biệt nhóm 0-6 tháng có TS = 0.5328). |
| `num_addon_services` | Numerical (`int64`) | `(df[['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)` (giá trị $[0, 6]$) | Định lượng độ gắn kết dịch vụ của khách hàng vào hệ sinh thái nhà mạng. |
| `avg_monthly_spend` | Numerical (`float64`) | `(df['TotalCharges'] / df['tenure'].replace(0, np.nan)).fillna(df['MonthlyCharges'])` | Đo lường áp lực tài chính thực tế mỗi tháng, khắc phục đa cộng tuyến giữa tenure và TotalCharges. |

### 5.2. Danh sách phân loại cột sau khi Fit (`params['num_cols']` và `params['cat_cols']`)

Từ kết quả fit trên tập Train, cấu trúc 22 đặc trưng đầu vào cho mô hình được xác định chính xác:

- **6 cột Numerical (`params['num_cols']`):**
  `['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'num_addon_services', 'avg_monthly_spend']`
- **16 cột Categorical (`params['cat_cols']`):**
  `['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']`
- **Shape sau biến đổi:**
  - $X_{\text{train}}$: $(5,638, 22)$, $y_{\text{train}}$: $(5,638,)$ với 0 giá trị NaN ở các cột Numerical.
  - Các cột Categorical được giữ nguyên định dạng chuỗi thô để thuật toán CatBoost tự xử lý qua Ordered Target Statistics nội bộ.
  - Tập validation (nếu cần) được tách từ tập train tại Notebook 04/05.
- **Tham số lưu trữ:**
  - File `data/processed/preprocessing_params.pkl` (kích thước 1.6 KB) chứa: danh sách cột (`num_cols`, `cat_cols`), giá trị median fit từ Train (`fill_params`), Target Statistics mapping (`ts_params` với `global_mean = 0.2653`, `smoothing = 1.0`).
  - Hệ số mất cân bằng: $\text{scale\_pos\_weight} = 2.7687$.

---

## 6. Kiểm soát Rò rỉ Dữ liệu (Data Leakage Control)

Hệ thống tuân thủ 4 nguyên tắc kiểm soát Data Leakage nghiêm ngặt được hiện thực hóa trực tiếp trong mã nguồn:

1. **Khử trùng lặp triệt để TRƯỚC khi Split (`01_split.ipynb`):**
   Toàn bộ các dòng trùng lặp hoàn toàn (47 dòng) và xung đột `customerID` (30 dòng) đều được giải quyết dứt điểm trước khi phân chia dữ liệu. Nếu split trước khi dedup, các bản ghi giống nhau có thể rơi đồng thời vào cả tập Train và Test, khiến mô hình học vẹt nhãn Test.
2. **Mọi tham số thống kê chỉ được Fit trên tập TRAIN:**
   Các giá trị trung vị (`median` để fill missing numerical) và bảng mã hóa Target Statistics (`ts_params`, `global_mean`) chỉ được tính toán duy nhất trên $5,638$ dòng của tập Train. Tập Validation và Test tuyệt đối không tham gia vào quá trình tính toán tham số.
3. **Tập Test được cô lập hoàn toàn (`test_raw.csv` không được mở ở bước 02 và 03):**
   Trong suốt quá trình EDA (`02_eda.ipynb`) và Feature Engineering (`03_feature_engineering.ipynb`), file `test_raw.csv` không hề được đọc vào bộ nhớ. Tập Test chỉ được nạp và transform duy nhất 1 lần tại bước đánh giá cuối cùng ở `04_train_model.ipynb`.
4. **Hàm `transform()` và `engineer_features()` là các hàm thuần (Stateless Transformations):**
   `engineer_features()` chỉ thực hiện các phép toán trên từng dòng dữ liệu độc lập; `transform()` chỉ áp dụng các tham số tĩnh đã được nạp từ `preprocessing_params.pkl` mà không học thêm bất kỳ thông tin nào từ dữ liệu truyền vào. Do đó, việc gọi transform trên tập Test ở bước sau hoàn toàn không phát sinh rò rỉ dữ liệu mới.

