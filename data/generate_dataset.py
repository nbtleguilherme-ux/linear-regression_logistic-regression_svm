"""Tạo lại tập dữ liệu nhà đất tổng hợp nha_dat.csv.

Chạy: python data/generate_dataset.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 320

dien_tich = np.random.randint(40, 220, N)
so_phong  = np.random.choice([1, 2, 3, 4, 5], N, p=[0.08, 0.22, 0.38, 0.22, 0.10])
kc_trung_tam = np.round(np.random.uniform(0.5, 35, N), 1)
tuoi_nha  = np.random.randint(0, 50, N)

# Giá (đơn vị: triệu đồng)
# Công thức thực: gia = 1500 + 12*dien_tich + 150*so_phong - 25*kc - 8*tuoi + noise
noise = np.random.normal(0, 180, N)
gia   = (1500
         + 12  * dien_tich
         + 150 * so_phong
         - 25  * kc_trung_tam
         - 8   * tuoi_nha
         + noise)
gia = np.maximum(gia, 400)       # giá tối thiểu 400 triệu
gia = np.round(gia, 1)

df = pd.DataFrame({
    'dien_tich':          dien_tich,
    'so_phong':           so_phong,
    'kc_trung_tam':       kc_trung_tam,
    'tuoi_nha':           tuoi_nha,
    'gia':                gia,
})

df.to_csv('data/nha_dat.csv', index=False)
print(f"Đã lưu {len(df)} dòng vào data/nha_dat.csv")
print(df.describe().round(1))
