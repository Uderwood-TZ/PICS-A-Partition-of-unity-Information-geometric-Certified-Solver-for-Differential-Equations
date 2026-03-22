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

DELTA = 0.018
L_SMOOTH = 0.02
A_T = 0.8
NU0 = 0.01
KAPPA = 0.01
EPS = 1.0
ALPHA_E = 0.50
SIGMA_J = 0.05
T0 = 300.0
T1 = 20.0

U0 = 1.0
P0 = 0.3
PHI0 = 0.7

W_C = 1.0
W_U = 1.0
W_V = 1.0
W_PHI = 1.0
W_T = 1.0
W_BC = 10.0

GRAD_CLIP = 5.0
INTERIOR_POOL = 65536
INIT_POOL_BATCH = 4096
DOMAIN_AREA = 1.0
BOUNDARY_LENGTH = 4.0
PLOT_DPI = 300


def set_global_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MLPNet(nn.Module):


    def __init__(self, in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH):
        super().__init__()
        layers = [nn.Linear(in_dim, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, xy):
        return self.net(xy)


def xs_centerline(y):
    return 0.5 + 0.12 * (y - 0.5) + 0.08 * torch.sin(2.0 * np.pi * y)


def dxs_dy(y):
    return 0.12 + 0.16 * np.pi * torch.cos(2.0 * np.pi * y)


def thin_layer_variables(x, y):
    xs = xs_centerline(y)
    z = (x - xs) / DELTA
    sS = torch.tanh(z)
    sech2 = 1.0 - sS * sS
    sS_x = sech2 / DELTA
    sS_y = -dxs_dy(y) * sech2 / DELTA
    return sS, sS_x, sS_y


def g_hot(x, y):
    return torch.exp(-((x - 0.25) ** 2 + (y - 0.75) ** 2) / 0.02)


def g_cold(x, y):
    return torch.exp(-((x - 0.78) ** 2 + (y - 0.28) ** 2) / 0.018)


def manufactured_fields(xy):
    x = xy[:, 0:1]
    y = xy[:, 1:2]
    sS, sS_x, sS_y = thin_layer_variables(x, y)
    hot = g_hot(x, y)
    cold = g_cold(x, y)

    u = 0.10 * np.pi * torch.sin(np.pi * x) * torch.cos(np.pi * y)
    u = u + 0.16 * (sS_y * torch.sin(np.pi * y) + sS * np.pi * torch.cos(np.pi * y))
    u = u + 0.10 * np.pi * torch.sin(2.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)
    u = u + 0.02 * np.pi * torch.cos(3.0 * np.pi * x) * torch.cos(np.pi * y)

    v = -0.10 * np.pi * torch.cos(np.pi * x) * torch.sin(np.pi * y)
    v = v - 0.16 * sS_x * torch.sin(np.pi * y)
    v = v - 0.10 * np.pi * torch.cos(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    v = v + 0.06 * np.pi * torch.sin(3.0 * np.pi * x) * torch.sin(np.pi * y)

    p = 1.0 + 0.24 * sS + 0.05 * torch.cos(2.0 * np.pi * y)
    p = p + 0.012 * torch.sin(2.0 * np.pi * x + 0.6 * np.pi * y)

    phi = 0.36 * torch.sin(2.0 * np.pi * x) * torch.sin(2.0 * np.pi * y)
    phi = phi + 0.16 * sS * torch.cos(np.pi * y)
    phi = phi + 0.02 * torch.sin(3.0 * np.pi * x) * torch.sin(np.pi * y)
    phi = phi + 0.012 * torch.cos(2.0 * np.pi * x + 0.6 * np.pi * y)

    T = 300.0 + 12.0 * sS
    T = T + 6.0 * torch.sin(2.0 * np.pi * x) * torch.cos(2.0 * np.pi * y)
    T = T + 3.0 * hot - 2.5 * cold
    T = T + 0.6 * torch.sin(2.0 * np.pi * x + 0.6 * np.pi * y)

    return {"u": u, "v": v, "p": p, "phi": phi, "T": T}


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


def divergence_of_flux(fx, fy, xy):
    fx_x = grad_scalar(fx, xy)[:, 0:1]
    fy_y = grad_scalar(fy, xy)[:, 1:2]
    return fx_x + fy_y


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

    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy
    ubar = u + 0.5 * (L_SMOOTH ** 2) * lap_u
    vbar = v + 0.5 * (L_SMOOTH ** 2) * lap_v

    T_tilde = (T - T0) / T1
    nu_T = NU0 * torch.exp(-A_T * T_tilde)
    diff_u = divergence_of_flux(nu_T * u_x, nu_T * u_y, xy_req)
    diff_v = divergence_of_flux(nu_T * v_x, nu_T * v_y, xy_req)

    f_u = ubar * u_x + vbar * u_y + p_x - diff_u + ALPHA_E * rho_e * phi_x
    f_v = ubar * v_x + vbar * v_y + p_y - diff_v + ALPHA_E * rho_e * phi_y
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


def drm_functional_terms(model, xy, rho_e, f_u, f_v, f_T):
    xy_req = xy.clone().detach().requires_grad_(True)
    pred = scale_outputs(model(xy_req))

    u = pred["u"]
    v = pred["v"]
    p = pred["p"]
    phi = pred["phi"]
    T = pred["T"]

    u_x, u_y, u_xx, u_yy = second_partials(u, xy_req)
    v_x, v_y, v_xx, v_yy = second_partials(v, xy_req)
    p_x, p_y, _, _ = second_partials(p, xy_req)
    phi_x, phi_y, phi_xx, phi_yy = second_partials(phi, xy_req)
    T_x, T_y, T_xx, T_yy = second_partials(T, xy_req)

    r_c = u_x + v_y

    T_tilde_pred = (T - T0) / T1
    nu_T_pred = NU0 * torch.exp(-A_T * T_tilde_pred)

    lap_u_pred = u_xx + u_yy
    lap_v_pred = v_xx + v_yy
    ubar_pred = u + 0.5 * (L_SMOOTH ** 2) * lap_u_pred
    vbar_pred = v + 0.5 * (L_SMOOTH ** 2) * lap_v_pred

    diff_u_pred = divergence_of_flux(nu_T_pred * u_x, nu_T_pred * u_y, xy_req)
    diff_v_pred = divergence_of_flux(nu_T_pred * v_x, nu_T_pred * v_y, xy_req)

    r_u = ubar_pred * u_x + vbar_pred * u_y + p_x - diff_u_pred + ALPHA_E * rho_e * phi_x - f_u
    r_v = ubar_pred * v_x + vbar_pred * v_y + p_y - diff_v_pred + ALPHA_E * rho_e * phi_y - f_v
    r_phi = -EPS * (phi_xx + phi_yy) - rho_e
    r_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y) - f_T

    return {
        "J_c": DOMAIN_AREA * W_C * torch.mean(r_c * r_c),
        "J_u": DOMAIN_AREA * W_U * torch.mean(r_u * r_u),
        "J_v": DOMAIN_AREA * W_V * torch.mean(r_v * r_v),
        "J_phi": DOMAIN_AREA * W_PHI * torch.mean(r_phi * r_phi),
        "J_T": DOMAIN_AREA * W_T * torch.mean(r_T * r_T),
    }


def boundary_functional(model, xy_b, truth_b):
    pred = scale_outputs(model(xy_b))
    penalty_density = (pred["u"] - truth_b["u"]) ** 2
    penalty_density = penalty_density + (pred["v"] - truth_b["v"]) ** 2
    penalty_density = penalty_density + (pred["p"] - truth_b["p"]) ** 2
    penalty_density = penalty_density + (pred["phi"] - truth_b["phi"]) ** 2
    penalty_density = penalty_density + (pred["T"] - truth_b["T"]) ** 2
    return BOUNDARY_LENGTH * torch.mean(penalty_density)


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
    lines.append("case09_drm baseline")
    lines.append("This script implements a Deep Ritz Method baseline for the same Case09 manufactured-solution PDE used in the previous Case09 workflow.")
    lines.append("The manufactured truth enters only through known source terms, boundary Dirichlet values, and post-training evaluation.")
    lines.append("interior truth supervision = 0")
    lines.append("optimizer = Adam only")
    lines.append("learning rate = fixed 1e-3")
    lines.append("training steps = 30000")
    lines.append("early stopping = none")
    lines.append("This implementation does not force a classical scalar Deep Ritz energy for the coupled Leray-smoothed, temperature-dependent-viscosity PDE system.")
    lines.append("Instead, it uses a least-squares variational Ritz baseline with Monte Carlo quadrature over the domain and the boundary.")
    lines.append("It remains on the variational or energy-minimization route and does not use pointwise adaptive PICS mechanisms.")
    lines.append("Case09 enters the variational functional through Leray-smoothed convection and temperature-dependent viscosity:")
    lines.append("ubar = u + 0.5*L_SMOOTH^2*(u_xx + u_yy), vbar = v + 0.5*L_SMOOTH^2*(v_xx + v_yy), nu_T = NU0*exp(-A_T*(T-T0)/T1)")
    lines.append("diff_u = d/dx(nu_T*u_x) + d/dy(nu_T*u_y), diff_v = d/dx(nu_T*v_x) + d/dy(nu_T*v_y)")
    lines.append("Variational functional:")
    lines.append("J_total = J_domain + W_BC * J_bc")
    lines.append("J_domain = Integral_Omega [w_c*r_c^2 + w_u*r_u^2 + w_v*r_v^2 + w_phi*r_phi^2 + w_T*r_T^2] dOmega")
    lines.append("J_bc = Integral_boundaryOmega [(u-u_star)^2 + (v-v_star)^2 + (p-p_star)^2 + (phi-phi_star)^2 + (T-T_star)^2] dGamma")
    lines.append("Numerically, the script approximates the domain integral by area-weighted Monte Carlo sampling on Omega and approximates the boundary integral by total-boundary-length-weighted Monte Carlo sampling on partial Omega.")
    lines.append("The script uses fixed affine output scaling to align the output convention with the Case09 baseline family.")
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
    lines.append(f"w_c = {W_C}")
    lines.append(f"w_u = {W_U}")
    lines.append(f"w_v = {W_V}")
    lines.append(f"w_phi = {W_PHI}")
    lines.append(f"w_T = {W_T}")
    lines.append(f"W_BC = {W_BC}")
    lines.append(f"grad_clip = {GRAD_CLIP}")
    lines.append(f"interior_pool = {INTERIOR_POOL}")
    lines.append(f"domain_area = {DOMAIN_AREA}")
    lines.append(f"boundary_length = {BOUNDARY_LENGTH}")
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
        (2, "J_total"),
        (3, "J_domain"),
        (4, "J_bc"),
        (5, "J_c"),
        (6, "J_u"),
        (7, "J_v"),
        (8, "J_phi"),
        (9, "J_T"),
    ]

    plt.figure(figsize=(8.0, 6.0))
    for idx, label in labels:
        plt.plot(steps, data[:, idx], label=label, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("functional")
    plt.title("loss linear")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_linear.png"), dpi=PLOT_DPI)
    plt.close()

    plt.figure(figsize=(8.0, 6.0))
    for idx, label in labels:
        plt.semilogy(steps, np.maximum(data[:, idx], 1.0e-30), label=label, linewidth=1.0)
    plt.xlabel("step")
    plt.ylabel("functional")
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
    parser = argparse.ArgumentParser(description="Least-squares variational DRM baseline for Case09 manufactured PDE system.")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    parser.add_argument("--outdir", type=str, default="case09_drm")
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

    write_readme(os.path.join(logs_dir, "README_case09_drm.txt"), args)

    model = MLPNet(in_dim=2, out_dim=5, width=WIDTH, depth=DEPTH).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    xy_pool_dict = build_interior_pool(device, dtype)
    xy_b_all, truth_b_all = boundary_points(device, dtype)

    history = np.zeros((MAX_STEPS, 10), dtype=np.float64)

    train_start = time.time()
    for step in range(1, MAX_STEPS + 1):
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
        domain_terms = drm_functional_terms(model, xy_f, rho_e, f_u, f_v, f_T)
        J_domain = domain_terms["J_c"] + domain_terms["J_u"] + domain_terms["J_v"] + domain_terms["J_phi"] + domain_terms["J_T"]
        J_bc = boundary_functional(model, xy_b, truth_b)
        J_total = J_domain + W_BC * J_bc
        J_total.backward()
        if GRAD_CLIP is not None and GRAD_CLIP > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        history[step - 1, 0] = float(step)
        history[step - 1, 1] = float(LR_INIT)
        history[step - 1, 2] = float(J_total.detach().cpu().item())
        history[step - 1, 3] = float(J_domain.detach().cpu().item())
        history[step - 1, 4] = float(J_bc.detach().cpu().item())
        history[step - 1, 5] = float(domain_terms["J_c"].detach().cpu().item())
        history[step - 1, 6] = float(domain_terms["J_u"].detach().cpu().item())
        history[step - 1, 7] = float(domain_terms["J_v"].detach().cpu().item())
        history[step - 1, 8] = float(domain_terms["J_phi"].detach().cpu().item())
        history[step - 1, 9] = float(domain_terms["J_T"].detach().cpu().item())

        print(
            "step={:05d} lr={:.6e} J_total={:.6e} J_domain={:.6e} J_bc={:.6e} J_c={:.6e} J_u={:.6e} J_v={:.6e} J_phi={:.6e} J_T={:.6e}".format(
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
        header="step lr J_total J_domain J_bc J_c J_u J_v J_phi J_T",
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

    with open(os.path.join(logs_dir, "metrics_case09_drm.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(metric_lines))

    with open(os.path.join(logs_dir, "runtime_case09_drm.txt"), "w", encoding="utf-8") as f:
        f.write(f"train_time_s {train_time:.16e}\n")
        f.write(f"eval_time_s {eval_time:.16e}\n")
        f.write(f"total_time_s {total_time:.16e}\n")

    plot_losses(loss_history_path, figs_dir)
    print_file_tree(base_dir)


if __name__ == "__main__":
    main()
