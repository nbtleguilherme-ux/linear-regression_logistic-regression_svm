"""Máy vector hỗ trợ (Support Vector Machine — SVM)."""
import numpy as np
import matplotlib.pyplot as plt


def huan_luyen_svm(X_train, y_train, C=1.0, kernel='linear'):
    """Huấn luyện mô hình SVM.

    Tham số:
        X_train : ma trận đặc trưng tập huấn luyện, shape (n, p)
        y_train : nhãn lớp, shape (n,)
        C       : tham số điều chuẩn (regularization)
                    - C nhỏ → siêu phẳng "mềm", margin rộng hơn, chấp nhận lỗi nhiều hơn
                    - C lớn → siêu phẳng "cứng", margin hẹp hơn, cố phân loại đúng tối đa
        kernel  : 'linear' hoặc 'rbf'

    Trả về:
        model : đối tượng SVC đã được fit

    Gợi ý: from sklearn.svm import SVC
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm huan_luyen_svm()")


def thong_tin_support_vectors(model):
    """Lấy thông tin về support vectors từ mô hình SVM đã huấn luyện.

    Tham số:
        model : đối tượng SVC đã được fit

    Trả về:
        dict với keys:
            'so_luong'    : tổng số support vectors (int)
            'moi_lop'     : số support vectors theo từng lớp (list)
            'toa_do'      : tọa độ support vectors (ndarray)

    Gợi ý: model.support_vectors_  và  model.n_support_
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm thong_tin_support_vectors()")


def ve_bien_quyet_dinh(model, X, y, ten_lop=None, tieu_de='SVM — Đường biên quyết định'):
    """Vẽ đường biên quyết định của SVM trên không gian 2 chiều.

    ⚠️ Hàm này chỉ hoạt động khi X có đúng 2 đặc trưng.

    Các bước gợi ý:
        1. Xác định phạm vi hiển thị: min/max của mỗi đặc trưng ± 0.5
        2. Tạo meshgrid với np.meshgrid
        3. Dự đoán nhãn cho từng điểm trên lưới
        4. Dùng plt.contourf để tô màu vùng phân loại
        5. Dùng plt.scatter để vẽ điểm dữ liệu
        6. Đánh dấu support vectors bằng ký hiệu '*'

    Tham số:
        model    : đối tượng SVC đã được fit
        X        : ma trận đặc trưng 2D, shape (n, 2)
        y        : nhãn, shape (n,)
        ten_lop  : tên các lớp (tùy chọn)
        tieu_de  : tiêu đề biểu đồ
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm ve_bien_quyet_dinh()")


def danh_gia_svm(model, X_test, y_test, ten_lop=None):
    """Đánh giá mô hình SVM trên tập kiểm tra.

    Tham số:
        model    : đối tượng SVC đã được fit
        X_test   : ma trận đặc trưng tập kiểm tra
        y_test   : nhãn thực tập kiểm tra
        ten_lop  : tên các lớp (tùy chọn)

    Trả về:
        dict với keys: 'accuracy', 'report', 'confusion_matrix'

    Gợi ý: sklearn.metrics — accuracy_score, classification_report, confusion_matrix
    """
    raise NotImplementedError("TODO: Hãy hoàn thiện hàm danh_gia_svm()")
