import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def chia_train_test(X, y, ty_le_test=0.2, random_state=42):
    """Chia dữ liệu thành tập huấn luyện và kiểm tra.

    Tham số:
        X            : ma trận đặc trưng, shape (n, p)
        y            : vector nhãn/giá trị, shape (n,)
        ty_le_test   : tỉ lệ dữ liệu kiểm tra, mặc định 0.2 (20 %)
        random_state : seed ngẫu nhiên để tái tạo kết quả

    Trả về:
        X_train, X_test, y_train, y_test

    Gợi ý: sklearn.model_selection.train_test_split
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm chia_train_test()")


def chuan_hoa(X_train, X_test):
    """Chuẩn hóa đặc trưng bằng StandardScaler (z-score normalization).

    ⚠️ Fit scaler CHỈ trên X_train rồi mới transform cả hai tập.
       Nếu fit trên X_test → data leakage (rò rỉ dữ liệu).

    Tham số:
        X_train : ma trận đặc trưng tập huấn luyện
        X_test  : ma trận đặc trưng tập kiểm tra

    Trả về:
        scaler      : đối tượng StandardScaler đã fit
        X_train_sc  : X_train sau khi chuẩn hóa
        X_test_sc   : X_test sau khi chuẩn hóa (dùng thống kê từ train)

    Gợi ý: sklearn.preprocessing.StandardScaler
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm chuan_hoa()")
