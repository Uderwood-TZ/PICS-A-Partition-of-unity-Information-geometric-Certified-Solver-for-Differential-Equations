
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
NU0 = 0.01
KAPPA = 0.01
EPS = 1.0
ALPHA_E = 0.50
SIGMA_J = 0.05
C_P = 0.0015
KAPPA_D = 8.0

U0 = 1.0
P0 = 0.3
PHI0 = 0.7
T0 = 300.0
T1 = 20.0

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


class DGMLayer(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.x_z = nn.Linear(in_dim, hidden_dim)
        self.s_z = nn.Linear(hidden_dim, hidden_dim)
        self.x_g = nn.Linear(in_dim, hidden_dim)
        self.s_g = nn.Linear(hidden_dim, hidden_dim)
        self.x_r = nn.Linear(in_dim, hidden_dim)
        self.s_r = nn.Linear(hidden_dim, hidden_dim)
        self.x_h = nn.Linear(in_dim, hidden_dim)
        self.s_h = nn.Linear(hidden_dim, hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        modules = [
            self.x_z, self.s_z, self.x_g, self.s_g,
            self.x_r, self.s_r, self.x_h, self.s_h,
        ]
        for module in modules:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x, s):
        z = torch.sigmoid(self.x_z(x) + self.s_z(s))
        g = torch.sigmoid(self.x_g(x) + self.s_g(s))
        r = torch.sigmoid(self.x_r(x) + self.s_r(s))
        h = torch.tanh(self.x_h(x) + self.s_h(r * s))
        s_new = (1.0 - g) * h + z * s
        return s_new


class DGMNet(nn.Module):


    def __init__(self, in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, width)
        self.hidden_layers = nn.ModuleList([DGMLayer(in_dim, width) for _ in range(depth)])
        self.output_layer = nn.Linear(width, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, xy):
        s = torch.tanh(self.input_layer(xy))
        for layer in self.hidden_layers:
            s = layer(xy, s)
        return self.output_layer(s)


def sx_variable(x, y):
    center_x = 0.45 + 0.03 * torch.sin(2.0 * np.pi * y)
    z = (x - center_x) / DELTA
    sx = torch.tanh(z)
    sech2 = 1.0 - sx * sx
    dsx_dx = sech2 / DELTA
    dsx_dy = -(0.06 * np.pi * torch.cos(2.0 * np.pi * y)) * sech2 / DELTA
    return sx, dsx_dx, dsx_dy


def sy_variable(x, y):
    center_y = 0.58 + 0.02 * torch.cos(2.0 * np.pi * x)
    z = (y - center_y) / DELTA
    sy = torch.tanh(z)
    sech2 = 1.0 - sy * sy
    dsy_dx = (0.04 * np.pi * torch.sin(2.0 * np.pi * x)) * sech2 / DELTA
    dsy_dy = sech2 / DELTA
    return sy, dsy_dx, dsy_dy


def combo_variable(x, y):
    sx, dsx_dx, dsx_dy = sx_variable(x, y)
    sy, dsy_dx, dsy_dy = sy_variable(x, y)
    phase = 2.0 * np.pi * x + 0.8 * np.pi * y
    s_combo = 0.55 * sx + 0.45 * sy + 0.08 * torch.sin(phase)
    dscombo_dx = 0.55 * dsx_dx + 0.45 * dsy_dx + 0.16 * np.pi * torch.cos(phase)
    dscombo_dy = 0.55 * dsx_dy + 0.45 * dsy_dy + 0.064 * np.pi * torch.cos(phase)
    return s_combo, dscombo_dx, dscombo_dy


def manufactured_fields(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    s_combo, dscombo_dx, dscombo_dy = combo_variable(x, y)
    phase = 2.0 * np.pi * x + 0.8 * np.pi * y

    psi = 0.10 * torch.sin(np.pi * x) * torch.sin(np.pi * y)
    psi = psi + 0.14 * s_combo * torch.sin(np.pi * y)
    psi = psi + 0.06 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    psi = psi + 0.02 * torch.cos(3.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)

    u = 0.10 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    u = u + 0.14 * (dscombo_dy * torch.sin(np.pi * y) + s_combo * np.pi * torch.cos(np.pi * y))
    u = u + 0.12 * np.pi * torch.sin(2.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)
    u = u + 0.04 * np.pi * torch.cos(3.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)

    v = -0.10 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    v = v - 0.14 * dscombo_dx * torch.sin(np.pi * y)
    v = v - 0.12 * np.pi * torch.cos(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    v = v + 0.06 * np.pi * torch.sin(3.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)

    p = 1.0 + 0.21 * s_combo + 0.05 * torch.cos(2.0 * np.pi * y)
    p = p + 0.012 * torch.sin(phase)

    phi = 0.34 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    phi = phi + 0.15 * s_combo * torch.cos(np.pi * y)
    phi = phi + 0.02 * torch.sin(3.0 * np.pi * x) * torch.sin(np.pi * y)
    phi = phi + 0.012 * torch.cos(phase)

    T = 300.0 + 12.0 * s_combo
    T = T + 7.0 * torch.sin(2.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)
    T = T + 1.2 * torch.cos(np.pi * x) * torch.sin(3.0 * np.pi * y)
    T = T + 0.9 * torch.sin(phase)

    return {"psi": psi, "u": u, "v": v, "p": p, "phi": phi, "T": T}


def grad_scalar(field, xy):
    return torch.autograd.grad(
        field,
        xy,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
    )[0]


def second_partials(field, xy):
    grad_field = grad_scalar(field, xy)
    dfdx = grad_field[:, 0:1]
    dfdy = grad_field[:, 1:2]
    d2fdx2 = torch.autograd.grad(
        dfdx,
        xy,
        grad_outputs=torch.ones_like(dfdx),
        create_graph=True,
        retain_graph=True,
    )[0][:, 0:1]
    d2fdy2 = torch.autograd.grad(
        dfdy,
        xy,
        grad_outputs=torch.ones_like(dfdy),
        create_graph=True,
        retain_graph=True,
    )[0][:, 1:2]
    return dfdx, dfdy, d2fdx2, d2fdy2


def pressure_derivative_terms(field, xy):
    grad_field = grad_scalar(field, xy)
    p_x = grad_field[:, 0:1]
    p_y = grad_field[:, 1:2]

    grad_px = grad_scalar(p_x, xy)
    p_xx = grad_px[:, 0:1]
    p_xy = grad_px[:, 1:2]

    grad_py = grad_scalar(p_y, xy)
    p_yx = grad_py[:, 0:1]
    p_yy = grad_py[:, 1:2]

    p_xxx = grad_scalar(p_xx, xy)[:, 0:1]
    p_xyy = grad_scalar(p_xy, xy)[:, 1:2]
    p_yxx = grad_scalar(p_yx, xy)[:, 0:1]
    p_yyy = grad_scalar(p_yy, xy)[:, 1:2]

    return {
        "p_x": p_x,
        "p_y": p_y,
        "p_xx": p_xx,
        "p_yy": p_yy,
        "p_xxx": p_xxx,
        "p_xyy": p_xyy,
        "p_yxx": p_yxx,
        "p_yyy": p_yyy,
    }


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
    p_terms = pressure_derivative_terms(p, xy_req)
    phi_x, phi_y, phi_xx, phi_yy = second_partials(phi, xy_req)
    T_x, T_y, T_xx, T_yy = second_partials(T, xy_req)

    rho_e = -EPS * (phi_xx + phi_yy) + (KAPPA_D ** 2) * phi

    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy
    lap_px = p_terms["p_xxx"] + p_terms["p_xyy"]
    lap_py = p_terms["p_yxx"] + p_terms["p_yyy"]

    f_u = u * u_x + v * u_y + p_terms["p_x"] - NU0 * lap_u + ALPHA_E * rho_e * phi_x + C_P * lap_px
    f_v = u * v_x + v * v_y + p_terms["p_y"] - NU0 * lap_v + ALPHA_E * rho_e * phi_y + C_P * lap_py
    f_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y)

    return {
        "rho_e": rho_e.detach(),
        "f_u": f_u.detach(),
        "f_v": f_v.detach(),
        "f_T": f_T.detach(),
    }


def scale_outputs(raw):
    u_tilde = raw[:, 0:1]
    v_tilde = raw[:, 1:2]
    p_tilde = raw[:, 2:3]
    phi_tilde = raw[:, 3:4]
    T_tilde = raw[:, 4:5]

    u = U0 * u_tilde
    v = U0 * v_tilde
    p = 1.0 + P0 * p_tilde
    phi = PHI0 * phi_tilde
    T = T0 + T1 * T_tilde
    return {"u": u, "v": v, "p": p, "phi": phi, "T": T}


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


def dgm_residual_losses(model, xy, rho_e, f_u, f_v, f_T):
    xy_req = xy.clone().detach().requires_grad_(True)
    pred = scale_outputs(model(xy_req))

    u = pred["u"]
    v = pred["v"]
    p = pred["p"]
    phi = pred["phi"]
    T = pred["T"]

    u_x, u_y, u_xx, u_yy = second_partials(u, xy_req)
    v_x, v_y, v_xx, v_yy = second_partials(v, xy_req)
    p_terms = pressure_derivative_terms(p, xy_req)
    phi_x, phi_y, phi_xx, phi_yy = second_partials(phi, xy_req)
    T_x, T_y, T_xx, T_yy = second_partials(T, xy_req)

    r_c = u_x + v_y
    r_u = u * u_x + v * u_y + p_terms["p_x"] - NU0 * (u_xx + u_yy) + ALPHA_E * rho_e * phi_x + C_P * (p_terms["p_xxx"] + p_terms["p_xyy"]) - f_u
    r_v = u * v_x + v * v_y + p_terms["p_y"] - NU0 * (v_xx + v_yy) + ALPHA_E * rho_e * phi_y + C_P * (p_terms["p_yxx"] + p_terms["p_yyy"]) - f_v
    r_phi = -EPS * (phi_xx + phi_yy) + (KAPPA_D ** 2) * phi - rho_e
    r_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y) - f_T

    return {
        "L_c": torch.mean(r_c * r_c),
        "L_u": torch.mean(r_u * r_u),
        "L_v": torch.mean(r_v * r_v),
        "L_phi": torch.mean(r_phi * r_phi),
        "L_T": torch.mean(r_T * r_T),
    }


def boundary_loss(model, xy_b, truth_b):
    pred = scale_outputs(model(xy_b))
    loss_u = torch.mean((pred["u"] - truth_b["u"]) ** 2)
    loss_v = torch.mean((pred["v"] - truth_b["v"]) ** 2)
    loss_p = torch.mean((pred["p"] - truth_b["p"]) ** 2)
    loss_phi = torch.mean((pred["phi"] - truth_b["phi"]) ** 2)
    loss_T = torch.mean((pred["T"] - truth_b["T"]) ** 2)
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
            pred_batch = scale_outputs(model(xy_batch))
            true_batch = manufactured_fields(xy_batch)
            for key in pred:
                pred[key].append(pred_batch[key].detach().cpu().numpy())
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
    lines.append("case10_dgm baseline")
    lines.append("This script implements an original-style Deep Galerkin Method baseline for the same Case10 manufactured-solution PDE used in the previous Case10 workflow.")
    lines.append("The manufactured truth enters only through known source terms, boundary Dirichlet values, and post-training evaluation.")
    lines.append("interior truth supervision = 0")
    lines.append("optimizer = Adam only")
    lines.append("training steps = 30000")
    lines.append("early stopping = none")
    lines.append("DGM principle: the solver uses a gated recurrent deep architecture to approximate PDE solution fields and directly minimizes strong-form PDE residuals together with boundary-condition mismatch.")
    lines.append("This script keeps only the original DGM solving idea and does not include MoE, gate routing by geometry, certificate logic, replay buffers, adaptive point insertion, band priors, or any current-method-specific enhancement.")
    lines.append("The current implementation keeps the Case10 screened-Poisson correction and pressure-gradient regularization.")
    lines.append("Specifically, rho_e = -EPS*(phi_xx + phi_yy) + KAPPA_D^2*phi and the momentum equations contain C_P*(p_xxx + p_xyy) and C_P*(p_yxx + p_yyy).")
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
    lines.append(f"U0 = {U0}")
    lines.append(f"P0 = {P0}")
    lines.append(f"PHI0 = {PHI0}")
    lines.append(f"T0 = {T0}")
    lines.append(f"T1 = {T1}")
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
    parser = argparse.ArgumentParser(description="Original-style DGM baseline for Case10 manufactured PDE system.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    parser.add_argument("--outdir", type=str, default="case10_dgm")
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

    write_readme(os.path.join(logs_dir, "README_case10_dgm.txt"), args)

    model = DGMNet(in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH).to(device=device, dtype=dtype)
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
        phys_losses = dgm_residual_losses(model, xy_f, rho_e, f_u, f_v, f_T)
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

    field_meta = {
        "u": {"title": "u", "prefix": "u"},
        "v": {"title": "v", "prefix": "v"},
        "p": {"title": "p", "prefix": "p"},
        "phi": {"title": "phi", "prefix": "phi"},
        "T": {"title": "T", "prefix": "t"},
    }

    metric_lines = ["field MSE RMSE MAE relL2 MaxError"]
    for field_name in ["u", "v", "p", "phi", "T"]:
        prefix = field_meta[field_name]["prefix"]
        title_name = field_meta[field_name]["title"]

        true_vec = true_fields[field_name]
        pred_vec = pred_fields[field_name]
        err_vec = np.abs(pred_vec - true_vec)

        true_grid = true_vec.reshape(NY_EVAL, NX_EVAL)
        pred_grid = pred_vec.reshape(NY_EVAL, NX_EVAL)
        err_grid = err_vec.reshape(NY_EVAL, NX_EVAL)

        vmin = min(np.min(true_grid), np.min(pred_grid))
        vmax = max(np.max(true_grid), np.max(pred_grid))
        emax = np.max(err_grid)

        plot_field(os.path.join(figs_dir, f"{prefix}_true.png"), X, Y, true_grid, f"{title_name} true", vmin, vmax)
        plot_field(os.path.join(figs_dir, f"{prefix}_pred.png"), X, Y, pred_grid, f"{title_name} pred", vmin, vmax)
        plot_field(os.path.join(figs_dir, f"{prefix}_maxerror.png"), X, Y, err_grid, f"{title_name} maxerror", 0.0, emax)

        save_field_txt(os.path.join(data_dir, f"{prefix}_true.txt"), X, Y, true_grid)
        save_field_txt(os.path.join(data_dir, f"{prefix}_pred.txt"), X, Y, pred_grid)
        save_field_txt(os.path.join(data_dir, f"{prefix}_maxerror.txt"), X, Y, err_grid)

        mse, rmse, mae, rel_l2, max_err = compute_metrics(true_vec, pred_vec)
        metric_lines.append(f"{field_name} {mse:.16e} {rmse:.16e} {mae:.16e} {rel_l2:.16e} {max_err:.16e}")

    with open(os.path.join(logs_dir, "metrics_case10_dgm.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(metric_lines))

    with open(os.path.join(logs_dir, "runtime_case10_dgm.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_time_s {train_time:.16e}\n")
        f.write(f"eval_time_s {eval_time:.16e}\n")
        f.write(f"total_time_s {total_time:.16e}\n")

    plot_losses(loss_history_path, figs_dir)
    print_file_tree(base_dir)


if __name__ == "__main__":
    main()
