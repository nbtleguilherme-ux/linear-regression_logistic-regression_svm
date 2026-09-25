"""Hồi quy tuyến tính — Gradient Descent và SGD.

Quy ước ký hiệu:
  n   : số mẫu
  p   : số đặc trưng (SAU khi thêm bias, nên p = số đặc trưng gốc + 1)
  w   : vector trọng số shape (p,), trong đó w[0] là bias
  X   : ma trận đặc trưng shape (n, p), cột đầu tiên là toàn số 1 (bias)
  y   : vector giá trị thực shape (n,)
"""
import numpy as np


# ---------------------------------------------------------------------------
# Hàm tiện ích — đã cho sẵn, KHÔNG cần chỉnh sửa
# ---------------------------------------------------------------------------

def them_bias(X):
    """Chèn cột 1 vào cột đầu tiên của X (bias term).

    Ví dụ: X shape (n, 4)  →  kết quả shape (n, 5)
    """
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# Bài tập — hoàn thiện các hàm bên dưới
# ---------------------------------------------------------------------------

def du_doan(X, w):
    """Tính giá trị dự đoán ŷ = X @ w.

    Tham số:
        X : ma trận đặc trưng (đã thêm cột bias), shape (n, p)
        w : vector trọng số, shape (p,)

    Trả về:
        y_hat : vector dự đoán, shape (n,)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm du_doan()")


def tinh_mse(y_true, y_pred):
    """Tính Mean Squared Error.

    Công thức:  MSE = (1/n) * Σ (y_true − y_pred)²

    Tham số:
        y_true : giá trị thực, shape (n,)
        y_pred : giá trị dự đoán, shape (n,)

    Trả về:
        mse : float
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm tinh_mse()")


def tinh_gradient(X, y, w):
    """Tính gradient của MSE theo w.

    Công thức:  ∇w = (2/n) * Xᵀ (Xw − y)

    Tham số:
        X : ma trận đặc trưng (đã thêm cột bias), shape (n, p)
        y : vector giá trị thực, shape (n,)
        w : vector trọng số hiện tại, shape (p,)

    Trả về:
        grad : vector gradient, shape (p,)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm tinh_gradient()")


def gradient_descent(X, y, alpha=0.01, n_iter=1000):
    """Thuật toán Gradient Descent (Batch GD) cho hồi quy tuyến tính.

    Thuật toán:
        1. Khởi tạo w = vector 0
        2. Lặp n_iter lần:
              grad  = tinh_gradient(X, y, w)
              w     = w − alpha * grad
              Lưu MSE(y, du_doan(X, w)) vào ls_loss

    Tham số:
        X      : ma trận đặc trưng (đã thêm cột bias), shape (n, p)
        y      : vector giá trị thực, shape (n,)
        alpha  : learning rate (tốc độ học)
        n_iter : số vòng lặp

    Trả về:
        w       : vector trọng số tối ưu, shape (p,)
        ls_loss : danh sách MSE tại mỗi vòng lặp, length n_iter
    """
    n, p = X.shape
    w = np.zeros(p)
    ls_loss = []

    # TODO: Hoàn thiện vòng lặp Gradient Descent
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm gradient_descent()")

    return w, ls_loss


def sgd(X, y, alpha=0.01, n_epochs=50, random_state=42):
    """Thuật toán Stochastic Gradient Descent (SGD).

    Khác với Batch GD, ở mỗi bước SGD chỉ dùng MỘT mẫu ngẫu nhiên.

    Thuật toán (mỗi epoch):
        a. Xáo trộn chỉ số [0, 1, ..., n−1]
        b. Với mỗi chỉ số i:
              Xi   = X[i:i+1, :]         # shape (1, p)
              yi   = y[i:i+1]            # shape (1,)
              grad = tinh_gradient(Xi, yi, w)
              w    = w − alpha * grad
        c. Tính MSE toàn bộ và lưu vào ls_loss

    Tham số:
        X            : ma trận đặc trưng (đã thêm bias), shape (n, p)
        y            : vector giá trị thực, shape (n,)
        alpha        : learning rate
        n_epochs     : số epochs (mỗi epoch đi qua toàn bộ dữ liệu)
        random_state : seed ngẫu nhiên

    Trả về:
        w       : vector trọng số tối ưu, shape (p,)
        ls_loss : danh sách MSE cuối mỗi epoch, length n_epochs
    """
    n, p = X.shape
    w = np.zeros(p)
    ls_loss = []
    rng = np.random.default_rng(random_state)

    # TODO: Hoàn thiện vòng lặp SGD
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm sgd()")

    return w, ls_loss


def phuong_trinh_chuan(X, y):
    """Giải hồi quy tuyến tính bằng Phương trình chuẩn (Normal Equation).

    Công thức:  w* = (XᵀX)⁻¹ Xᵀ y

    Gợi ý: dùng np.linalg.pinv thay vì np.linalg.inv để tránh lỗi ma trận suy biến.

    Tham số:
        X : ma trận đặc trưng (đã thêm cột bias), shape (n, p)
        y : vector giá trị thực, shape (n,)

    Trả về:
        w : vector trọng số giải tích, shape (p,)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm phuong_trinh_chuan()")
