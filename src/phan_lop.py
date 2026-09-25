"""Phân lớp tuyến tính — Hồi quy Logistic (Binary & Multi-class).

Quy ước ký hiệu:
  n   : số mẫu
  p   : số đặc trưng (bao gồm bias)
  K   : số lớp
  w   : vector trọng số cho binary, shape (p,)
  W   : ma trận trọng số cho multi-class, shape (K, p)
"""
import numpy as np


# ---------------------------------------------------------------------------
# Hàm tiện ích — đã cho sẵn, KHÔNG cần chỉnh sửa
# ---------------------------------------------------------------------------

def them_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# PHẦN A — Hồi quy Logistic nhị phân (2 lớp)
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Hàm kích hoạt sigmoid: σ(z) = 1 / (1 + e^(−z)).

    Tham số:
        z : scalar hoặc ndarray bất kỳ hình dạng

    Trả về:
        Giá trị nằm trong khoảng (0, 1)

    Gợi ý: np.exp(−z)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm sigmoid()")


def xac_suat_binary(X, w):
    """Tính P(y=1 | X) = sigmoid(X @ w).

    Tham số:
        X : ma trận đặc trưng (đã thêm bias), shape (n, p)
        w : vector trọng số, shape (p,)

    Trả về:
        probs : xác suất thuộc lớp 1, shape (n,)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm xac_suat_binary()")


def du_doan_binary(X, w, nguong=0.5):
    """Phân loại nhị phân dựa trên ngưỡng xác suất.

    y_pred[i] = 1 nếu P(y=1|X[i]) ≥ nguong, ngược lại = 0

    Tham số:
        X      : ma trận đặc trưng (đã thêm bias), shape (n, p)
        w      : vector trọng số, shape (p,)
        nguong : ngưỡng phân loại, mặc định 0.5

    Trả về:
        y_pred : nhãn dự đoán, shape (n,), kiểu int
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm du_doan_binary()")


def log_loss(X, y, w):
    """Binary Cross-Entropy Loss (log loss).

    Công thức:  L = −(1/n) Σ [ y log(p) + (1−y) log(1−p) ]
    trong đó    p = sigmoid(X @ w)

    Gợi ý: np.clip(p, 1e-15, 1 − 1e-15) để tránh log(0)

    Tham số:
        X : ma trận đặc trưng (đã thêm bias), shape (n, p)
        y : nhãn thực (0 hoặc 1), shape (n,)
        w : vector trọng số, shape (p,)

    Trả về:
        loss : float
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm log_loss()")


def gradient_logistic(X, y, w):
    """Gradient của log loss theo w.

    Công thức:  ∇w = (1/n) Xᵀ (p − y)
    trong đó    p = sigmoid(X @ w)

    Tham số:
        X : ma trận đặc trưng (đã thêm bias), shape (n, p)
        y : nhãn thực (0 hoặc 1), shape (n,)
        w : vector trọng số hiện tại, shape (p,)

    Trả về:
        grad : vector gradient, shape (p,)
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm gradient_logistic()")


def huan_luyen_logistic(X, y, alpha=0.1, n_iter=1000):
    """Huấn luyện hồi quy logistic bằng Gradient Descent.

    Thuật toán:
        1. Khởi tạo w = vector 0
        2. Lặp n_iter lần:
              grad = gradient_logistic(X, y, w)
              w    = w − alpha * grad
              Lưu log_loss(X, y, w) vào ls_loss

    Tham số:
        X      : ma trận đặc trưng (đã thêm bias), shape (n, p)
        y      : nhãn thực (0 hoặc 1), shape (n,)
        alpha  : learning rate
        n_iter : số vòng lặp

    Trả về:
        w       : vector trọng số tối ưu, shape (p,)
        ls_loss : danh sách loss tại mỗi vòng lặp
    """
    n, p = X.shape
    w = np.zeros(p)
    ls_loss = []

    # TODO: Hoàn thiện vòng lặp huấn luyện
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm huan_luyen_logistic()")

    return w, ls_loss


# ---------------------------------------------------------------------------
# PHẦN B — Phân lớp nhiều lớp với sklearn (wrapper đơn giản)
# ---------------------------------------------------------------------------

def huan_luyen_da_lop(X_train, y_train, alpha=0.1, max_iter=1000):
    """Huấn luyện hồi quy logistic nhiều lớp (Softmax / One-vs-Rest).

    Tham số:
        X_train  : ma trận đặc trưng tập huấn luyện, shape (n, p)
        y_train  : nhãn lớp, shape (n,)  — giá trị nguyên 0, 1, 2, ...
        alpha    : learning rate (solver='saga')
        max_iter : số vòng lặp tối đa

    Trả về:
        model : đối tượng LogisticRegression đã được fit

    Gợi ý:
        from sklearn.linear_model import LogisticRegression
        Dùng multi_class='multinomial', solver='lbfgs', max_iter=max_iter
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm huan_luyen_da_lop()")


# ---------------------------------------------------------------------------
# PHẦN C — Đánh giá (dùng chung cho binary và multi-class)
# ---------------------------------------------------------------------------

def danh_gia(y_true, y_pred, ten_lop=None):
    """In và trả về các chỉ số đánh giá phân lớp.

    Tính: accuracy, classification_report, confusion_matrix.

    Tham số:
        y_true   : nhãn thực
        y_pred   : nhãn dự đoán
        ten_lop  : danh sách tên lớp (tùy chọn)

    Trả về:
        dict với keys: 'accuracy', 'report', 'confusion_matrix'

    Gợi ý: sklearn.metrics — accuracy_score, classification_report, confusion_matrix
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm danh_gia()")
