import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 把本脚本放到数据文件同一目录下直接运行
# 脚本会自动读取当前目录中的：
#   u_true / u_pred / u_maxerror
#   v_true / v_pred / v_maxerror
#   p_true / p_pred / p_maxerror
#   phi_true / phi_pred / phi_maxerror
#   t_true / t_pred / t_maxerror
# 支持文件名带 .txt，也支持 Windows 隐藏扩展名时看到的“无扩展名”情况
# 输出目录：当前目录/replot_pics_style
# 绘图风格：按 PICS 重绘
#   1) pcolormesh
#   2) cmap = jet
#   3) true 和 pred 共用 true 的色标范围
#   4) maxerror 单独用 [0, max(error)]
# ============================================================

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR
OUT_DIR = os.path.join(SCRIPT_DIR, "replot_pics_style")
os.makedirs(OUT_DIR, exist_ok=True)

FIELDS = ["u", "v", "p", "phi", "t"]
SUFFIXES = ["true", "pred", "maxerror"]


def find_file(stem):
    """
    在当前目录下查找：
    1) 完全匹配 stem 的文件
    2) stem + 任意扩展名 的文件
    """
    exact_path = os.path.join(DATA_DIR, stem)
    if os.path.isfile(exact_path):
        return exact_path

    matches = glob.glob(os.path.join(DATA_DIR, stem + ".*"))
    matches = [m for m in matches if os.path.isfile(m)]
    if matches:
        # 优先 txt
        txt_matches = [m for m in matches if m.lower().endswith(".txt")]
        if txt_matches:
            return txt_matches[0]
        return matches[0]

    raise FileNotFoundError(f"未找到文件：{stem} 或 {stem}.*")


def load_triplet(path):
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"文件格式不对，至少需要三列 x, y, value：{path}")
    x = arr[:, 0]
    y = arr[:, 1]
    v = arr[:, 2]
    return x, y, v


def triplet_to_grid(x, y, v):
    """
    这里不重新排序，直接按原导出顺序 reshape。
    因为原脚本通常使用 meshgrid 后再 reshape(-1) 导出，
    只要三份文件来自同一套网格，这种恢复方式最稳。
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)

    if nx * ny != len(v):
        raise ValueError(
            f"网格点数不一致：nx={nx}, ny={ny}, nx*ny={nx*ny}, len(v)={len(v)}"
        )

    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    V = v.reshape(ny, nx)
    return X, Y, V


def plot_field(path_png, X, Y, V, title, vmin, vmax):
    fig = plt.figure(figsize=(6.2, 5.2), dpi=150)
    ax = fig.add_subplot(111)
    im = ax.pcolormesh(X, Y, V, shading="auto", cmap="jet", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax)
    cb.ax.set_ylabel(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


def save_copy_txt(path_txt, X, Y, V):
    data = np.column_stack([X.reshape(-1), Y.reshape(-1), V.reshape(-1)])
    np.savetxt(path_txt, data, fmt="%.12e")


def main():
    report_lines = []
    report_lines.append("Replot started.")
    report_lines.append(f"DATA_DIR = {DATA_DIR}")
    report_lines.append(f"OUT_DIR  = {OUT_DIR}")
    report_lines.append("")

    for fld in FIELDS:
        true_path = find_file(f"{fld}_true")
        pred_path = find_file(f"{fld}_pred")
        err_path  = find_file(f"{fld}_maxerror")

        xt, yt, vt = load_triplet(true_path)
        xp, yp, vp = load_triplet(pred_path)
        xe, ye, ve = load_triplet(err_path)

        X,  Y,  Vtrue = triplet_to_grid(xt, yt, vt)
        Xp, Yp, Vpred = triplet_to_grid(xp, yp, vp)
        Xe, Ye, Verr  = triplet_to_grid(xe, ye, ve)

        # 基本一致性检查
        if not (X.shape == Xp.shape == Xe.shape and Y.shape == Yp.shape == Ye.shape):
            raise ValueError(f"{fld} 的 true/pred/error 网格形状不一致。")

        if not (np.allclose(X, Xp) and np.allclose(X, Xe) and np.allclose(Y, Yp) and np.allclose(Y, Ye)):
            raise ValueError(f"{fld} 的 true/pred/error 坐标网格不一致。")

        vmin = float(np.min(Vtrue))
        vmax = float(np.max(Vtrue))
        emax = float(np.max(Verr))

        title_true = f"{fld} true"
        title_pred = f"{fld} pred"
        title_err  = f"{fld} maxerror"

        plot_field(os.path.join(OUT_DIR, f"{fld}_true.png"), X, Y, Vtrue, title_true, vmin, vmax)
        plot_field(os.path.join(OUT_DIR, f"{fld}_pred.png"), X, Y, Vpred, title_pred, vmin, vmax)
        plot_field(os.path.join(OUT_DIR, f"{fld}_maxerror.png"), X, Y, Verr, title_err, 0.0, emax)

        # 顺手把按当前重排后的 txt 再存一份，方便核对
        save_copy_txt(os.path.join(OUT_DIR, f"{fld}_true.txt"), X, Y, Vtrue)
        save_copy_txt(os.path.join(OUT_DIR, f"{fld}_pred.txt"), X, Y, Vpred)
        save_copy_txt(os.path.join(OUT_DIR, f"{fld}_maxerror.txt"), X, Y, Verr)

        report_lines.append(f"[OK] {fld}")
        report_lines.append(f"  true file = {os.path.basename(true_path)}")
        report_lines.append(f"  pred file = {os.path.basename(pred_path)}")
        report_lines.append(f"  err  file = {os.path.basename(err_path)}")
        report_lines.append(f"  shape     = {X.shape}")
        report_lines.append(f"  true min/max = {vmin:.12e}, {vmax:.12e}")
        report_lines.append(f"  err  max     = {emax:.12e}")
        report_lines.append("")

    report_lines.append("Replot finished.")
    with open(os.path.join(OUT_DIR, "replot_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
