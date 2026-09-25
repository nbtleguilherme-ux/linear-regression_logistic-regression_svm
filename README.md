# Buổi 10 — Các Mô hình Tuyến tính

Thực hành cho **Chương 18: Các Mô hình Tuyến tính** (khoảng 3 giờ).

## Cấu trúc dự án

```
session10/
├── data/
│   ├── nha_dat.csv          ← Tập dữ liệu giá nhà (320 mẫu, 4 đặc trưng)
│   └── generate_dataset.py  ← Script tái tạo CSV
├── notebooks/
│   ├── 01_HoiQuy_TuyenTinh.ipynb    ← (~90 phút) Gradient Descent, SGD
│   ├── 02_PhanLop_TuyenTinh.ipynb   ← (~60 phút) Logistic Regression
│   └── 03_SVM.ipynb                 ← (~30 phút) Support Vector Machine
├── src/
│   ├── preprocessing.py   ← Chia dữ liệu, chuẩn hóa
│   ├── hoi_quy.py         ← Gradient Descent, SGD, Normal Equation
│   ├── phan_lop.py        ← Logistic Regression (binary + multi-class)
│   └── svm_utils.py       ← SVM, vẽ đường biên, đánh giá
└── requirements.txt
```

## Cách bắt đầu

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_HoiQuy_TuyenTinh.ipynb
```

## Nội dung từng notebook

| Notebook | Chủ đề | Hàm cần hoàn thiện |
|---|---|---|
| 01 | Hồi quy tuyến tính | `du_doan`, `tinh_mse`, `tinh_gradient`, `gradient_descent`, `sgd`, `phuong_trinh_chuan` |
| 02 | Phân lớp tuyến tính | `sigmoid`, `xac_suat_binary`, `log_loss`, `gradient_logistic`, `huan_luyen_logistic`, `du_doan_binary`, `huan_luyen_da_lop`, `danh_gia` |
| 03 | SVM | `huan_luyen_svm`, `thong_tin_support_vectors`, `ve_bien_quyet_dinh`, `danh_gia_svm` |

## Quy trình làm bài

1. Mở notebook theo thứ tự (01 → 02 → 03).
2. Đọc hướng dẫn trong từng ô Markdown.
3. Mở file `src/` tương ứng, hoàn thiện hàm bị đánh dấu `raise NotImplementedError(...)`.
4. Chạy lại ô code trong notebook để kiểm tra.
5. Trả lời các câu hỏi phân tích cuối mỗi phần.

## Dữ liệu — nha_dat.csv

Dữ liệu tổng hợp mô phỏng giá nhà tại một thành phố nhỏ:

| Cột | Ý nghĩa | Đơn vị |
|---|---|---|
| `dien_tich` | Diện tích sàn | m² |
| `so_phong` | Số phòng ngủ | — |
| `kc_trung_tam` | Khoảng cách đến trung tâm | km |
| `tuoi_nha` | Tuổi căn nhà | năm |
| `gia` | **Biến mục tiêu** — Giá nhà | triệu đồng |
