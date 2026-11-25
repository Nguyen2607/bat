"""Thuật toán dơi (Bat Algorithm) cho bài toán chia giờ ôn thi - cài đặt theo mục 2.2 và 2.4 của nhóm.
4 môn, tổng quỹ thời gian 30 giờ, mỗi môn có giờ tối thiểu. Bài toán CỰC ĐẠI hoá điểm TB có trọng số tín chỉ.
Chạy:  python bat_chia_gio.py
"""
import numpy as np

MON    = ["Giải tích", "Lập trình Python", "Vật lý Đại cương", "Triết học Mác - Lênin"]
TIN_CHI = np.array([4.0, 3.0, 3.0, 2.0])        # c_j
K_KHO   = np.array([0.12, 0.18, 0.10, 0.22])    # k_j
H_MIN   = np.array([4.0, 3.0, 3.0, 2.0])        # h_j,min
T_TONG  = 30.0                                  # tổng quỹ thời gian


def diem_mon(h, k=K_KHO):
    """Điểm dự kiến mỗi môn: S_j = 10 (1 - exp(-k_j h_j))."""
    return 10.0 * (1.0 - np.exp(-k * np.asarray(h)))


def F(h, c=TIN_CHI, k=K_KHO):
    """Hàm mục tiêu: điểm TB có trọng số tín chỉ (cần cực đại)."""
    return float(np.sum(c * diem_mon(h, k)) / np.sum(c))


def anh_xa(x, h_min=H_MIN, t_tong=T_TONG):
    """Kỹ thuật phân bổ tỉ lệ (mục 2.4.3): h_j = h_min_j + T_dư * |x_j| / sum|x_k|.
    Luôn thoả sum(h) = t_tong và h_j >= h_min_j nên không cần hàm phạt."""
    a = np.abs(np.asarray(x, dtype=float)) + 1e-12
    return h_min + (t_tong - h_min.sum()) * a / a.sum()


def nghiem_giai_tich(t_tong=T_TONG, h_min=H_MIN, c=TIN_CHI, k=K_KHO):
    """Nghiệm tham chiếu chính xác (điều kiện KKT, tìm hệ số Lagrange bằng chia đôi)."""
    a = c * 10.0 * k
    h_of = lambda lam: np.maximum(h_min, np.log(a / lam) / k)
    lo, hi = 1e-12, a.max()
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        if h_of(mid).sum() > t_tong: lo = mid
        else: hi = mid
    return h_of(np.sqrt(lo * hi))


def mien_tim_kiem(t_tong=T_TONG, h_min=H_MIN):
    """Miền của vector vị trí x: [h_min_j, T - tổng giờ tối thiểu của các môn còn lại]."""
    return h_min.copy(), t_tong - (h_min.sum() - h_min)


def bat_algorithm(f_muc_tieu=None, lb=None, ub=None, n=30, t_max=200, alpha=0.9, gamma=0.9,
                  f_min=0.0, f_max=2.0, A0=1.5, r0=0.5, sigma=None, seed=None, **kw):
    """BA cho bài toán cực đại hoá, theo mã giả mục 2.2.4 (20 dòng). Trả về (x*, F*, lịch sử F*, số lần đánh giá)."""
    rng = np.random.default_rng(seed)
    if lb is None: lb, ub = mien_tim_kiem(kw.get("t_tong", T_TONG), kw.get("h_min", H_MIN))
    if f_muc_tieu is None:
        t_tong, h_min = kw.get("t_tong", T_TONG), kw.get("h_min", H_MIN)
        f_muc_tieu = lambda x: F(anh_xa(x, h_min, t_tong))
    if sigma is None: sigma = 0.1 * float(np.mean(ub - lb))        # σ ≈ 0,1 × bề rộng miền (ghi chú mục 2.2.3)
    D = len(lb)
    x = lb + rng.random((n, D)) * (ub - lb)                        # dòng 1
    v = np.zeros((n, D)); A = np.full(n, A0); r = np.full(n, r0)   # dòng 1-2
    fx = np.array([f_muc_tieu(xi) for xi in x])                    # dòng 3
    k = fx.argmax(); best, f_best = x[k].copy(), fx[k]
    lich_su = [f_best]; evals = n
    for t in range(1, t_max + 1):                                  # dòng 5-6
        A_bar = A.mean()                                           # dòng 7, công thức (2.7)
        for i in range(n):                                         # dòng 8
            f_i = f_min + (f_max - f_min) * rng.random()           # dòng 9, (2.1)
            v[i] = v[i] + (x[i] - best) * f_i                      # dòng 10, (2.2)
            x_new = x[i] + v[i]                                    # dòng 11, (2.3)
            if rng.random() > r[i]:                                # dòng 12
                x_new = best + sigma * rng.uniform(-1, 1, D) * A_bar   # dòng 13, (2.4)
            x_new = np.clip(x_new, lb, ub)                         # dòng 14: chặn biên (chuẩn hoá giờ nằm trong anh_xa)
            f_new = f_muc_tieu(x_new); evals += 1                  # dòng 15
            if rng.random() < A[i] and f_new > fx[i]:              # dòng 16
                x[i], fx[i] = x_new, f_new                         # dòng 17
                A[i] = alpha * A[i]; r[i] = r0 * (1.0 - np.exp(-gamma * t))   # dòng 18, (2.5)-(2.6)
                if fx[i] > f_best: best, f_best = x[i].copy(), fx[i]          # dòng 19
        lich_su.append(f_best)
    return best, f_best, np.array(lich_su), evals                  # dòng 20


if __name__ == "__main__":
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x, f, ls, ev = bat_algorithm(seed=1)
    h_ba, h_gt, h_deu = anh_xa(x), nghiem_giai_tich(), np.full(4, T_TONG / 4)
    print(f"{'Môn':<24}{'BA':>8}{'Giải tích':>11}{'Chia đều':>10}")
    for j, m in enumerate(MON): print(f"{m:<24}{h_ba[j]:8.2f}{h_gt[j]:11.2f}{h_deu[j]:10.2f}")
    print(f"{'Tổng giờ':<24}{h_ba.sum():8.2f}{h_gt.sum():11.2f}{h_deu.sum():10.2f}")
    print(f"{'Điểm TB (thang 10)':<24}{F(h_ba):8.4f}{F(h_gt):11.4f}{F(h_deu):10.4f}   (số lần đánh giá hàm: {ev})")
    xs = np.arange(4); w = 0.27
    plt.figure(figsize=(7.5, 4))
    plt.bar(xs - w, h_deu, w, label="Chia đều"); plt.bar(xs, h_ba, w, label="BA"); plt.bar(xs + w, h_gt, w, label="Nghiệm giải tích")
    plt.xticks(xs, MON, fontsize=8); plt.ylabel("Số giờ"); plt.legend(); plt.tight_layout(); plt.savefig("demo_phan_bo.png", dpi=150)
    print("Đã lưu demo_phan_bo.png")
