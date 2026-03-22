import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

SEED_DEFAULT = 1234
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_DEFAULT = torch.float64
LR_INIT = 1.0e-3
MAX_STEPS = 30000
WIDTH = 64
DEPTH = 4
ACTIVATION = "tanh"
BATCH_F = 1024
BATCH_B = 1024
EVAL_BATCH = 4096
NX_EVAL = 201
NY_EVAL = 201
N_EDGE = 1000

DELTA = 0.02
NU = 0.01
KAPPA = 0.01
EPS = 1.0
ALPHA_E = 0.50
SIGMA_J = 0.05

W_BC = 10.0
GRAD_CLIP = 5.0
LR_FINAL_RATIO = 0.1
INTERIOR_POOL = 65536
INIT_POOL_BATCH = 4096
PLOT_DPI = 300


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPNet(nn.Module):
    """
    Vanilla PINN backbone.

    A single plain MLP maps (x, y) directly to the five physical fields
    (u, v, p, phi, T). The model does not use streamfunction parameterization,
    hard constraints, gating, expert mixtures, Fourier features, or any other
    enhanced mechanism.
    """

    def __init__(self, in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers += [nn.Linear(width, out_dim)]
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy):
        return self.net(xy)


def x0_centerline(y):
    return 0.5 + 0.03 * torch.sin(2.0 * np.pi * y) + 0.01 * torch.sin(6.0 * np.pi * y)


def x0_centerline_y(y):
    return 0.06 * np.pi * torch.cos(2.0 * np.pi * y) + 0.06 * np.pi * torch.cos(6.0 * np.pi * y)


def thin_layer_variables(x, y):
    x0 = x0_centerline(y)
    z = (x - x0) / DELTA
    s = torch.tanh(z)
    sech2 = 1.0 - s * s
    s_x = sech2 / DELTA
    s_y = -sech2 * x0_centerline_y(y) / DELTA
    return s, s_x, s_y


def manufactured_fields(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    s, s_x, s_y = thin_layer_variables(x, y)

    u = 0.10 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    u = u + 0.20 * (s_y * torch.sin(np.pi * y) + s * np.pi * torch.cos(np.pi * y))

    v = -0.10 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    v = v - 0.20 * s_x * torch.sin(np.pi * y)

    p = 1.0 + 0.30 * s + 0.05 * torch.cos(2.0 * np.pi * y)
    p = p + 0.01 * torch.sin(2.0 * np.pi * x) + 0.008 * torch.sin(np.pi * x) * torch.sin(3.0 * np.pi * y)

    phi = 0.50 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    phi = phi + 0.20 * s * torch.cos(np.pi * y) + 0.02 * torch.sin(3.0 * np.pi * x) * torch.sin(np.pi * y)

    T = 300.0 + 20.0 * s + 5.0 * torch.sin(2.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)
    T = T + 1.5 * torch.cos(np.pi * x) * torch.sin(3.0 * np.pi * y)

    return {"u": u, "v": v, "p": p, "phi": phi, "T": T}


def grad_scalar(field, xy):
    return torch.autograd.grad(field, xy, grad_outputs=torch.ones_like(field), create_graph=True, retain_graph=True)[0]


def second_partials(field, xy):
    grad_field = grad_scalar(field, xy)
    dfdx = grad_field[:, 0:1]
    dfdy = grad_field[:, 1:2]
    d2fdx2 = torch.autograd.grad(dfdx, xy, grad_outputs=torch.ones_like(dfdx), create_graph=True, retain_graph=True)[0][:, 0:1]
    d2fdy2 = torch.autograd.grad(dfdy, xy, grad_outputs=torch.ones_like(dfdy), create_graph=True, retain_graph=True)[0][:, 1:2]
    return dfdx, dfdy, d2fdx2, d2fdy2


def manufactured_sources(xy):
    xy_req = xy.clone().detach().requires_grad_(True)
    fields = manufactured_fields(xy_req)

    u = fields["u"]
    v = fields["v"]
    p = fields["p"]
    phi = fields["phi"]
    T = fields["T"]

    u_x, u_y, u_xx, u_yy = second_partials(u, xy_req)
    v_x, v_y, v_xx, v_yy = second_partials(v, xy_req)
    p_x, p_y, _, _ = second_partials(p, xy_req)
    phi_x, phi_y, phi_xx, phi_yy = second_partials(phi, xy_req)
    T_x, T_y, T_xx, T_yy = second_partials(T, xy_req)

    rho_e = -EPS * (phi_xx + phi_yy)
    f_u = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + ALPHA_E * rho_e * phi_x
    f_v = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + ALPHA_E * rho_e * phi_y
    f_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y)

    return {
        "rho_e": rho_e.detach(),
        "f_u": f_u.detach(),
        "f_v": f_v.detach(),
        "f_T": f_T.detach(),
    }


def uniform_interior_pool(n_points, device, dtype):
    xy = np.random.rand(n_points, 2)
    return torch.tensor(xy, dtype=dtype, device=device)


def boundary_points(device, dtype):
    grid = torch.linspace(0.0, 1.0, N_EDGE, dtype=dtype, device=device).view(-1, 1)
    zeros = torch.zeros_like(grid)
    ones = torch.ones_like(grid)

    bottom = torch.cat([grid, zeros], dim=1)
    top = torch.cat([grid, ones], dim=1)
    left = torch.cat([zeros, grid], dim=1)
    right = torch.cat([ones, grid], dim=1)

    xy_b = torch.cat([bottom, top, left, right], dim=0)
    truth_b = manufactured_fields(xy_b)
    return xy_b, truth_b


def batch_indices(n_total, batch_size):
    if batch_size >= n_total:
        return np.random.randint(0, n_total, size=batch_size)
    return np.random.choice(n_total, size=batch_size, replace=False)


def build_interior_pool(device, dtype):
    xy_pool = uniform_interior_pool(INTERIOR_POOL, device, dtype)
    rho_pool = torch.zeros((INTERIOR_POOL, 1), dtype=dtype, device=device)
    fu_pool = torch.zeros((INTERIOR_POOL, 1), dtype=dtype, device=device)
    fv_pool = torch.zeros((INTERIOR_POOL, 1), dtype=dtype, device=device)
    fT_pool = torch.zeros((INTERIOR_POOL, 1), dtype=dtype, device=device)

    start = 0
    while start < INTERIOR_POOL:
        end = min(start + INIT_POOL_BATCH, INTERIOR_POOL)
        src = manufactured_sources(xy_pool[start:end])
        rho_pool[start:end] = src["rho_e"]
        fu_pool[start:end] = src["f_u"]
        fv_pool[start:end] = src["f_v"]
        fT_pool[start:end] = src["f_T"]
        start = end

    return {
        "xy": xy_pool,
        "rho_e": rho_pool,
        "f_u": fu_pool,
        "f_v": fv_pool,
        "f_T": fT_pool,
    }


def pinn_residual_losses(model, xy, rho_e, f_u, f_v, f_T):
    xy_req = xy.clone().detach().requires_grad_(True)
    raw = model(xy_req)
    u = raw[:, 0:1]
    v = raw[:, 1:2]
    p = raw[:, 2:3]
    phi = raw[:, 3:4]
    T = raw[:, 4:5]

    u_x, u_y, u_xx, u_yy = second_partials(u, xy_req)
    v_x, v_y, v_xx, v_yy = second_partials(v, xy_req)
    p_x, p_y, _, _ = second_partials(p, xy_req)
    phi_x, phi_y, phi_xx, phi_yy = second_partials(phi, xy_req)
    T_x, T_y, T_xx, T_yy = second_partials(T, xy_req)

    r_c = u_x + v_y
    r_u = u * u_x + v * u_y + p_x - NU * (u_xx + u_yy) + ALPHA_E * rho_e * phi_x - f_u
    r_v = u * v_x + v * v_y + p_y - NU * (v_xx + v_yy) + ALPHA_E * rho_e * phi_y - f_v
    r_phi = -EPS * (phi_xx + phi_yy) - rho_e
    r_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y) - f_T

    return {
        "L_c": torch.mean(r_c * r_c),
        "L_u": torch.mean(r_u * r_u),
        "L_v": torch.mean(r_v * r_v),
        "L_phi": torch.mean(r_phi * r_phi),
        "L_T": torch.mean(r_T * r_T),
    }


def boundary_loss(model, xy_b, truth_b):
    pred = model(xy_b)
    u = pred[:, 0:1]
    v = pred[:, 1:2]
    p = pred[:, 2:3]
    phi = pred[:, 3:4]
    T = pred[:, 4:5]

    loss_u = torch.mean((u - truth_b["u"]) ** 2)
    loss_v = torch.mean((v - truth_b["v"]) ** 2)
    loss_p = torch.mean((p - truth_b["p"]) ** 2)
    loss_phi = torch.mean((phi - truth_b["phi"]) ** 2)
    loss_T = torch.mean((T - truth_b["T"]) ** 2)
    return loss_u + loss_v + loss_p + loss_phi + loss_T


def update_learning_rate(optimizer, step):
    if MAX_STEPS <= 1:
        lr = LR_INIT
    else:
        cosine = 0.5 * (1.0 + np.cos(np.pi * (step - 1) / (MAX_STEPS - 1)))
        lr = LR_INIT * (LR_FINAL_RATIO + (1.0 - LR_FINAL_RATIO) * cosine)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def evaluate_model(model, xy, device, dtype):
    model.eval()
    n_total = xy.shape[0]
    pred = {"u": [], "v": [], "p": [], "phi": [], "T": []}
    true = {"u": [], "v": [], "p": [], "phi": [], "T": []}
    with torch.no_grad():
        for start in range(0, n_total, EVAL_BATCH):
            end = min(start + EVAL_BATCH, n_total)
            xy_batch = xy[start:end].to(device=device, dtype=dtype)
            pred_batch = model(xy_batch)
            pred["u"].append(pred_batch[:, 0:1].detach().cpu().numpy())
            pred["v"].append(pred_batch[:, 1:2].detach().cpu().numpy())
            pred["p"].append(pred_batch[:, 2:3].detach().cpu().numpy())
            pred["phi"].append(pred_batch[:, 3:4].detach().cpu().numpy())
            pred["T"].append(pred_batch[:, 4:5].detach().cpu().numpy())
            true_batch = manufactured_fields(xy_batch)
            for key in true:
                true[key].append(true_batch[key].detach().cpu().numpy())
    for key in pred:
        pred[key] = np.vstack(pred[key]).reshape(-1)
        true[key] = np.vstack(true[key]).reshape(-1)
    model.train()
    return pred, true


def save_field_txt(path, X, Y, V):
    data = np.column_stack([X.reshape(-1), Y.reshape(-1), V.reshape(-1)])
    np.savetxt(path, data, fmt="%.16e")


def plot_field(path, X, Y, V, title, vmin, vmax):
    plt.figure(figsize=(6.0, 5.0))
    im = plt.imshow(
        V,
        extent=[float(X.min()), float(X.max()), float(Y.min()), float(Y.max())],
        origin="lower",
        cmap="jet",
        aspect="auto",
        vmin=vmin,
        vmax=vmax,
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(path, dpi=PLOT_DPI)
    plt.close()


def compute_metrics(true_vec, pred_vec):
    err = pred_vec - true_vec
    mse = np.mean(err ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(err))
    denom = np.sqrt(np.sum(true_vec ** 2))
    rel_l2 = np.sqrt(np.sum(err ** 2)) / denom if denom > 0.0 else np.nan
    max_err = np.max(np.abs(err))
    return mse, rmse, mae, rel_l2, max_err


def write_readme(path, args):
    lines = []
    lines.append("case01_pinn baseline")
    lines.append("This script implements the most original vanilla PINN baseline for the same manufactured-solution PDE case used in the previous case.")
    lines.append("The manufactured truth enters only through source terms, boundary Dirichlet values, and post-training evaluation.")
    lines.append("interior truth supervision = 0")
    lines.append("optimizer = Adam only")
    lines.append("training steps = 30000")
    lines.append("early stopping = none")
    lines.append("pretraining = none")
    lines.append("L-BFGS = none")
    lines.append("The current PINN uses a single plain MLP and directly outputs u, v, p, phi, T.")
    lines.append("The continuity equation enters the loss as an ordinary residual term. The script does not use streamfunction hard constraints.")
    lines.append("No adaptive sampling, replay buffer, certificate, gate, MoE, Fourier feature, hard constraint transform, or interior supervised data is used.")
    lines.append("Loss function:")
    lines.append("L_total = L_phys + W_BC * L_bc")
    lines.append("L_phys = mean(r_c^2 + r_u^2 + r_v^2 + r_phi^2 + r_T^2)")
    lines.append("L_bc = MSE(u,u_star) + MSE(v,v_star) + MSE(p,p_star) + MSE(phi,phi_star) + MSE(T,T_star) on the boundary")
    lines.append("core hyperparameters:")
    lines.append(f"seed = {args.seed}")
    lines.append(f"device = {args.device}")
    lines.append("dtype = torch.float64")
    lines.append(f"lr_init = {LR_INIT}")
    lines.append(f"max_steps = {MAX_STEPS}")
    lines.append(f"width = {WIDTH}")
    lines.append(f"depth = {DEPTH}")
    lines.append(f"activation = {ACTIVATION}")
    lines.append(f"batch_f = {BATCH_F}")
    lines.append(f"batch_b = {BATCH_B}")
    lines.append(f"eval_batch = {EVAL_BATCH}")
    lines.append(f"nx_eval = {NX_EVAL}")
    lines.append(f"ny_eval = {NY_EVAL}")
    lines.append(f"n_edge = {N_EDGE}")
    lines.append(f"W_BC = {W_BC}")
    lines.append(f"grad_clip = {GRAD_CLIP}")
    lines.append(f"interior_pool = {INTERIOR_POOL}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def plot_losses(loss_history_path, figs_dir):
    data = np.loadtxt(loss_history_path, skiprows=1)
    steps = data[:, 0]
    labels = [
        (2, "L_total"),
        (3, "L_phys"),
        (4, "L_bc"),
        (5, "L_c"),
        (6, "L_u"),
        (7, "L_v"),
        (8, "L_phi"),
        (9, "L_T"),
    ]

    plt.figure(figsize=(8.0, 6.0))
    for idx, label in labels:
        plt.plot(steps, data[:, idx], label=label, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("loss linear")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_linear.png"), dpi=PLOT_DPI)
    plt.close()

    plt.figure(figsize=(8.0, 6.0))
    for idx, label in labels:
        plt.semilogy(steps, np.maximum(data[:, idx], 1.0e-30), label=label, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("loss log")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_log.png"), dpi=PLOT_DPI)
    plt.close()


def print_file_tree(root_dir):
    print("\nGenerated files:")
    for current_root, _, files in os.walk(root_dir):
        files = sorted(files)
        for filename in files:
            path = os.path.join(current_root, filename)
            rel = os.path.relpath(path, root_dir)
            print(rel)


def main():
    parser = argparse.ArgumentParser(description="Vanilla PINN baseline for case01 manufactured PDE system.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    parser.add_argument("--outdir", type=str, default="case01_pinn")
    args = parser.parse_args()

    set_global_seed(args.seed)
    torch.set_default_dtype(DTYPE_DEFAULT)

    device = torch.device(args.device)
    dtype = DTYPE_DEFAULT

    base_dir = os.path.abspath(args.outdir)
    figs_dir = os.path.join(base_dir, "figs")
    data_dir = os.path.join(base_dir, "data")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    write_readme(os.path.join(logs_dir, "README_case01_pinn.txt"), args)

    model = MLPNet(in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    xy_pool_dict = build_interior_pool(device, dtype)
    xy_b_all, truth_b_all = boundary_points(device, dtype)

    history = np.zeros((MAX_STEPS, 10), dtype=np.float64)

    train_start = time.time()
    for step in range(1, MAX_STEPS + 1):
        lr_now = update_learning_rate(optimizer, step)
        idx_f = batch_indices(INTERIOR_POOL, BATCH_F)
        idx_b = batch_indices(xy_b_all.shape[0], BATCH_B)

        xy_f = xy_pool_dict["xy"][idx_f]
        rho_e = xy_pool_dict["rho_e"][idx_f]
        f_u = xy_pool_dict["f_u"][idx_f]
        f_v = xy_pool_dict["f_v"][idx_f]
        f_T = xy_pool_dict["f_T"][idx_f]

        xy_b = xy_b_all[idx_b]
        truth_b = {key: truth_b_all[key][idx_b] for key in truth_b_all}

        optimizer.zero_grad(set_to_none=True)
        phys_losses = pinn_residual_losses(model, xy_f, rho_e, f_u, f_v, f_T)
        L_phys = phys_losses["L_c"] + phys_losses["L_u"] + phys_losses["L_v"] + phys_losses["L_phi"] + phys_losses["L_T"]
        L_bc = boundary_loss(model, xy_b, truth_b)
        L_total = L_phys + W_BC * L_bc
        L_total.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        history[step - 1, 0] = float(step)
        history[step - 1, 1] = float(lr_now)
        history[step - 1, 2] = float(L_total.detach().cpu().item())
        history[step - 1, 3] = float(L_phys.detach().cpu().item())
        history[step - 1, 4] = float(L_bc.detach().cpu().item())
        history[step - 1, 5] = float(phys_losses["L_c"].detach().cpu().item())
        history[step - 1, 6] = float(phys_losses["L_u"].detach().cpu().item())
        history[step - 1, 7] = float(phys_losses["L_v"].detach().cpu().item())
        history[step - 1, 8] = float(phys_losses["L_phi"].detach().cpu().item())
        history[step - 1, 9] = float(phys_losses["L_T"].detach().cpu().item())

        print(
            "step={:05d} lr={:.6e} L_total={:.6e} L_phys={:.6e} L_bc={:.6e} L_c={:.6e} L_u={:.6e} L_v={:.6e} L_phi={:.6e} L_T={:.6e}".format(
                step,
                history[step - 1, 1],
                history[step - 1, 2],
                history[step - 1, 3],
                history[step - 1, 4],
                history[step - 1, 5],
                history[step - 1, 6],
                history[step - 1, 7],
                history[step - 1, 8],
                history[step - 1, 9],
            ),
            flush=True,
        )
    train_time = time.time() - train_start

    loss_history_path = os.path.join(logs_dir, "loss_history.txt")
    np.savetxt(
        loss_history_path,
        history,
        fmt="%.16e",
        header="step lr L_total L_phys L_bc L_c L_u L_v L_phi L_T",
        comments="",
    )

    x_eval = np.linspace(0.0, 1.0, NX_EVAL)
    y_eval = np.linspace(0.0, 1.0, NY_EVAL)
    X, Y = np.meshgrid(x_eval, y_eval, indexing="xy")
    xy_eval = np.column_stack([X.reshape(-1), Y.reshape(-1)])
    xy_eval_torch = torch.tensor(xy_eval, dtype=dtype, device=device)

    eval_start = time.time()
    pred_fields, true_fields = evaluate_model(model, xy_eval_torch, device, dtype)
    eval_time = time.time() - eval_start
    total_time = train_time + eval_time

    metric_lines = ["field MSE RMSE MAE relL2 MaxError"]
    for field_name in ["u", "v", "p", "phi", "T"]:
        true_vec = true_fields[field_name]
        pred_vec = pred_fields[field_name]
        err_vec = np.abs(pred_vec - true_vec)

        true_grid = true_vec.reshape(NY_EVAL, NX_EVAL)
        pred_grid = pred_vec.reshape(NY_EVAL, NX_EVAL)
        err_grid = err_vec.reshape(NY_EVAL, NX_EVAL)

        vmin = min(np.min(true_grid), np.min(pred_grid))
        vmax = max(np.max(true_grid), np.max(pred_grid))
        emax = np.max(err_grid)

        plot_field(os.path.join(figs_dir, f"{field_name}_true.png"), X, Y, true_grid, f"{field_name} true", vmin, vmax)
        plot_field(os.path.join(figs_dir, f"{field_name}_pred.png"), X, Y, pred_grid, f"{field_name} pred", vmin, vmax)
        plot_field(os.path.join(figs_dir, f"{field_name}_maxerror.png"), X, Y, err_grid, f"{field_name} maxerror", 0.0, emax)

        save_field_txt(os.path.join(data_dir, f"{field_name}_true.txt"), X, Y, true_grid)
        save_field_txt(os.path.join(data_dir, f"{field_name}_pred.txt"), X, Y, pred_grid)
        save_field_txt(os.path.join(data_dir, f"{field_name}_maxerror.txt"), X, Y, err_grid)

        mse, rmse, mae, rel_l2, max_err = compute_metrics(true_vec, pred_vec)
        metric_lines.append(f"{field_name} {mse:.16e} {rmse:.16e} {mae:.16e} {rel_l2:.16e} {max_err:.16e}")

    with open(os.path.join(logs_dir, "metrics_case01_pinn.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(metric_lines))

    with open(os.path.join(logs_dir, "runtime_case01_pinn.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_time_s {train_time:.16e}\n")
        f.write(f"eval_time_s {eval_time:.16e}\n")
        f.write(f"total_time_s {total_time:.16e}\n")

    plot_losses(loss_history_path, figs_dir)
    print_file_tree(base_dir)


if __name__ == "__main__":
    main()
