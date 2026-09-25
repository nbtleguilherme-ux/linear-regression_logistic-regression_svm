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
    from sklearn.svm import SVC
    model = SVC(C=C, kernel=kernel, random_state=42)
    model.fit(X_train, y_train)
    return model


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
    return {
        "so_luong": int(model.support_vectors_.shape[0]),
        "moi_lop": list(model.n_support_),
        "toa_do": model.support_vectors_,
    }


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
    x0_min, x0_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x1_min, x1_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x0_min, x0_max, 300),
        np.linspace(x1_min, x1_max, 300),
    )
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.25, cmap="RdBu")

    colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    markers = ["o", "s", "^", "D"]
    for k, cls in enumerate(np.unique(y)):
        idx   = y == cls
        label = ten_lop[k] if ten_lop else str(cls)
        plt.scatter(
            X[idx, 0], X[idx, 1],
            c=colors[k % len(colors)],
            marker=markers[k % len(markers)],
            label=label,
            edgecolors="k", linewidths=0.3, s=40, alpha=0.85,
        )

    # Support vectors — vẽ viền đậm
    sv = model.support_vectors_
    plt.scatter(
        sv[:, 0], sv[:, 1],
        s=160, linewidths=1.8,
        facecolors="none", edgecolors="black",
        zorder=5, label="Support Vectors",
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title(tieu_de)
    plt.legend()
    plt.tight_layout()
    plt.show()


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
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, ConfusionMatrixDisplay)

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=ten_lop)
    cm     = confusion_matrix(y_test, y_pred)

    print(f"Độ chính xác: {acc:.4f}\n")
    print(report)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ten_lop).plot(
        ax=ax, colorbar=False, cmap="Blues"
    )
    ax.set_title("Ma trận nhầm lẫn")
    plt.tight_layout()
    plt.show()

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}
