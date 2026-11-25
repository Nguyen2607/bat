import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bat_chia_gio import *

RUNS = 30
H_OPT = nghiem_giai_tich(); OPT = F(H_OPT)
LB, UB = mien_tim_kiem()
gap = lambda f: max(OPT - f, 0.0)                       # chênh lệch điểm TB so với nghiệm chính xác (>= 0)
obj = lambda x: F(anh_xa(x))

def pso(n=30, t_max=200, w=0.7, c1=1.5, c2=1.5, seed=None):
    rng = np.random.default_rng(seed); D = len(LB)
    x = LB + rng.random((n, D)) * (UB - LB); v = np.zeros((n, D))
    fx = np.array([obj(a) for a in x]); pb, pf = x.copy(), fx.copy()
    k = pf.argmax(); gb, gf = pb[k].copy(), pf[k]; ls = [gf]
    for t in range(t_max):
        r1, r2 = rng.random((n, D)), rng.random((n, D))
        v = w * v + c1 * r1 * (pb - x) + c2 * r2 * (gb - x)
        x = np.clip(x + v, LB, UB); fx = np.array([obj(a) for a in x])
        m = fx > pf; pb[m], pf[m] = x[m], fx[m]; k = pf.argmax()
        if pf[k] > gf: gb, gf = pb[k].copy(), pf[k]
        ls.append(gf)
    return gb, gf, np.array(ls), n * (t_max + 1)

def ga(n=30, t_max=200, pc=0.9, pm=0.2, seed=None):
    rng = np.random.default_rng(seed); D = len(LB); w = UB - LB
    x = LB + rng.random((n, D)) * w; fx = np.array([obj(a) for a in x])
    k = fx.argmax(); gb, gf = x[k].copy(), fx[k]; ls = [gf]
    for t in range(t_max):
        def tour():
            i, j = rng.integers(0, n, 2); return x[i] if fx[i] > fx[j] else x[j]
        new = [gb.copy()]
        while len(new) < n:
            p1, p2 = tour(), tour()
            c = (lambda a: a * p1 + (1 - a) * p2)(rng.random(D)) if rng.random() < pc else p1.copy()
            c = np.clip(c + (rng.random(D) < pm) * rng.normal(0, 0.1, D) * w, LB, UB); new.append(c)
        x = np.array(new); fx = np.array([obj(a) for a in x]); k = fx.argmax()
        if fx[k] > gf: gb, gf = x[k].copy(), fx[k]
        ls.append(gf)
    return gb, gf, np.array(ls), n * (t_max + 1)

def ngau_nhien(evals=6030, seed=None):
    rng = np.random.default_rng(seed); best = -1e9
    for _ in range(evals): best = max(best, obj(LB + rng.random(len(LB)) * (UB - LB)))
    return None, best, None, evals

def st(g):
    a = np.array(g); return dict(best=float(a.min()), mean=float(a.mean()), std=float(a.std()), worst=float(a.max()), median=float(np.median(a)))

out = {"opt": OPT, "h_opt": H_OPT.tolist(), "evals": 6030, "sigma_default": 0.1 * float(np.mean(UB - LB)), "lb": LB.tolist(), "ub": UB.tolist()}
doc_h = np.array([9.25, 7.85, 8.15, 4.75])
out["check"] = {"ex1": F(np.array([8, 8, 8, 6.])), "S_ex1": diem_mon(np.array([8, 8, 8, 6.])).tolist(),
                "H_new": anh_xa([4, 4, 2, 1]).tolist(), "F_new": F(anh_xa([4, 4, 2, 1])),
                "H_old": anh_xa([5, 4, 3, 2]).tolist(), "F_old": F(anh_xa([5, 4, 3, 2])),
                "F_doc_ref": F(doc_h), "S_doc_ref": diem_mon(doc_h).tolist()}
out["base"] = {"deu": F(np.full(4, 7.5)), "tin_chi": F(H_MIN + 18 * TIN_CHI / TIN_CHI.sum()), "doc_ref": F(doc_h),
               "h_tin_chi": (H_MIN + 18 * TIN_CHI / TIN_CHI.sum()).tolist()}
runs = [bat_algorithm(seed=s) for s in range(RUNS)]
g1 = [gap(r[1]) for r in runs]; out["th1"] = st(g1); out["th1"]["F_mean"] = float(np.mean([r[1] for r in runs]))
out["th1"]["n_1e3"] = int(sum(g < 1e-3 for g in g1)); out["th1"]["n_1e4"] = int(sum(g < 1e-4 for g in g1))
bi = int(np.argmax([r[1] for r in runs])); out["h_ba"] = anh_xa(runs[bi][0]).tolist(); out["F_ba_best"] = float(runs[bi][1])
out["h_ba_all_max_dev"] = float(max(np.abs(anh_xa(r[0]) - H_OPT).max() for r in runs))
Hba = np.array([[gap(f) for f in r[2]] for r in runs])
out["conv_ba"] = {str(t): float(np.median(Hba[:, t])) for t in [0, 5, 10, 25, 50, 100, 200]}
def sweep(name, vals):
    rows = []
    for v in vals:
        g = [gap(bat_algorithm(seed=s, **{name: v})[1]) for s in range(RUNS)]
        rows.append(dict(value=v, **st(g)))
    return rows
out["sw_n"] = sweep("n", [10, 20, 30, 50]); out["sw_alpha"] = sweep("alpha", [0.5, 0.8, 0.9, 0.97])
out["sw_gamma"] = sweep("gamma", [0.1, 0.5, 0.9]); out["sw_sigma"] = sweep("sigma", [0.5, 1.8, 4.0, 8.0])
out["sw_tmax"] = sweep("t_max", [50, 100, 200, 400])
pr = [pso(seed=s) for s in range(RUNS)]; gr = [ga(seed=s) for s in range(RUNS)]; rr = [ngau_nhien(seed=s) for s in range(RUNS)]
gp, gg, gn = [gap(r[1]) for r in pr], [gap(r[1]) for r in gr], [gap(r[1]) for r in rr]
out["pso"], out["ga"], out["rand"] = st(gp), st(gg), st(gn)
Hp = np.array([[gap(f) for f in r[2]] for r in pr]); Hg = np.array([[gap(f) for f in r[2]] for r in gr])
sc = []
for T in [20, 30, 45]:
    ho = nghiem_giai_tich(T); opt = F(ho); lb, ub = mien_tim_kiem(T)
    rs = [bat_algorithm(seed=s, t_tong=T) for s in range(RUNS)]
    gg_ = [max(opt - r[1], 0) for r in rs]; b = int(np.argmax([r[1] for r in rs]))
    sc.append(dict(T=T, opt=opt, ba_mean=float(np.mean([r[1] for r in rs])), gap_worst=float(max(gg_)), deu=F(np.full(4, T / 4)),
                   h_opt=ho.tolist(), h_ba=anh_xa(rs[b][0], H_MIN, T).tolist()))
out["scen"] = sc
json.dump(out, open("results.json", "w"), indent=1, ensure_ascii=False)

# ---- hình ----
fl = lambda a: np.maximum(a, 1e-12)
plt.figure(figsize=(6.2, 3.8))
for H, lab in [(Hba, "BA"), (Hp, "PSO"), (Hg, "GA")]: plt.plot(fl(np.median(H, axis=0)), label=lab)
plt.yscale("log"); plt.xlabel("Vòng lặp"); plt.ylabel("Chênh lệch điểm TB so với tối ưu (trung vị)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_conv.png", dpi=200); plt.close()
xs = np.arange(4); w = 0.2; hb = np.array(out["h_ba"])
plt.figure(figsize=(6.6, 3.9))
plt.bar(xs - 1.5 * w, np.full(4, 7.5), w, label="Chia đều"); plt.bar(xs - 0.5 * w, out["base"]["h_tin_chi"], w, label="Chia theo tín chỉ")
plt.bar(xs + 0.5 * w, hb, w, label="BA"); plt.bar(xs + 1.5 * w, H_OPT, w, label="Nghiệm giải tích")
plt.xticks(xs, ["Giải tích", "Lập trình\nPython", "Vật lý\nĐại cương", "Triết học\nMác - Lênin"], fontsize=8); plt.ylabel("Số giờ"); plt.legend(fontsize=8); plt.grid(axis="y", alpha=.3)
plt.tight_layout(); plt.savefig("fig_alloc.png", dpi=200); plt.close()
plt.figure(figsize=(5.6, 3.7))
plt.boxplot([fl(np.array(a)) for a in [g1, gp, gg, gn]], tick_labels=["BA", "PSO", "GA", "Ngẫu nhiên"])
plt.yscale("log"); plt.ylabel("Chênh lệch điểm TB so với tối ưu"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_box.png", dpi=200); plt.close()
plt.figure(figsize=(5.2, 3.6)); s = out["sw_sigma"]
plt.errorbar([str(r["value"]) for r in s], [r["mean"] for r in s], yerr=[r["std"] for r in s], marker="o", capsize=3)
plt.xlabel("Hệ số bước cục bộ σ"); plt.ylabel("Δ trung bình ± độ lệch chuẩn"); plt.yscale("log"); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_sigma.png", dpi=200); plt.close()
print(json.dumps({k: out[k] for k in ["opt", "h_opt", "sigma_default", "check", "base", "th1", "conv_ba", "h_ba", "h_ba_all_max_dev", "pso", "ga", "rand"]}, indent=1)[:4500])
for k in ["sw_n", "sw_alpha", "sw_gamma", "sw_sigma", "sw_tmax"]:
    for r in out[k]: print(k, r["value"], "%.2e %.2e %.2e" % (r["mean"], r["std"], r["worst"]))
for s in sc: print(s["T"], round(s["opt"], 4), round(s["ba_mean"], 4), round(s["deu"], 4), s["gap_worst"], np.round(s["h_opt"], 2), np.round(s["h_ba"], 2))
