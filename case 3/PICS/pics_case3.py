

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ============================================================
# Hyperparameters (MUST be right below imports)
# ============================================================

SEED_DEFAULT = 1234
DEVICE_DEFAULT = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE_DEFAULT = torch.float64

LR_INIT = 1e-3
MAX_STEPS = 30000  # fixed, must run full
CERT_INTERVAL = 100

# Grids
NX_EVAL, NY_EVAL = 201, 201
NX_CERT, NY_CERT = 401, 401

# Sampling
NF_INIT = 20000
N_EDGE = 1000  # per edge; Nb=4*N_EDGE
BAND_W = 0.12
BAND_RATIO = 0.60  # 60% band, 40% uniform in regular sampling

# Hard buffer + topK
BUFFER_SIZE = 20000
TOPK_U = 200
TOPK_V = 200
TOPK_PHI = 200
TOPK_T = 200
TOPK_MAX = 400
ADD_BAND_PER_SCAN = 200  # extra band points added to interior pool every scan

# Batches
BATCH_F = 1024
BATCH_B = 1024
CERT_BATCH = 2048
EVAL_BATCH = 4096

# Batch mixture
BUF_RATIO_IN_BATCH = 0.70  # 70% buffer, 30% regular

# Loss weights
W_BC = 10.0
W_CERT = 1.0
W_INT = 0.1

# Certificate proxy
BETA_CERT = 20.0

# Stability
GRAD_CLIP_NORM = 5.0

# LR schedule
LR_FINAL_RATIO = 0.1  # cosine anneal to LR_INIT*LR_FINAL_RATIO
CERT_STAGNATION_WINDOW = 5
CERT_MIN_REL_DROP = 0.05
LR_HALVE_FACTOR = 0.5

# Gate / thin layers
DELTA_G = 0.02  # gating width

# Case10 manufactured thin layer
DELTA = 0.02

# Network
ACTIVATION = "tanh"   # "tanh" or "silu"
WIDTH = 64
DEPTH = 4  # number of hidden layers

# Physical params
NU0 = 0.01
KAPPA = 0.01
EPS = 1.0
ALPHA_E = 0.50
SIGMA_J = 0.05

# Case10 new correction terms
C_P = 0.0015      # pressure-gradient regularization coefficient
KAPPA_D = 8.0     # Debye-Hückel screening parameter

# Affine scales (constants only)
P0 = 0.3
PHI0 = 0.7
T0 = 300.0
T1 = 20.0
PSI0 = 1.0

# Record-only tol (no early stop)
TOL_CERT_RECORD = 1e-3

# ============================================================
# Core helpers
# ============================================================

torch.set_default_dtype(DTYPE_DEFAULT)
PI = np.pi


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_s() -> float:
    return time.time()


def rms(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(x * x))


def sech2(z: torch.Tensor) -> torch.Tensor:
    return 1.0 / torch.cosh(z).pow(2)


def grad(outputs: torch.Tensor, inputs: torch.Tensor, create_graph: bool) -> torch.Tensor:
    g = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
        only_inputs=True,
        allow_unused=False,
    )[0]
    return g


def chunked_range(n: int, bs: int):
    i = 0
    while i < n:
        j = min(n, i + bs)
        yield i, j
        i = j


def ascii_bar(i: int, n: int, width: int = 24) -> str:
    if n <= 0:
        return "[" + ("#" * width) + "]"
    r = max(0.0, min(1.0, float(i) / float(n)))
    k = int(round(r * width))
    return "[" + ("#" * k) + ("-" * (width - k)) + "]"


# ============================================================
# Gate + features (fixed, not trained, not truth)
# ============================================================

def x0_of_y(y: torch.Tensor) -> torch.Tensor:
    return 0.50 + 0.02 * torch.sin(2.0 * PI * y)


def gate_g(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x0 = x0_of_y(y)
    return 0.5 * (1.0 + torch.tanh((x - x0) / DELTA_G))


def feature_xi(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x0 = x0_of_y(y)
    return (x - x0) / DELTA_G


# ============================================================
# Case10 manufactured truth (ONLY for RHS, BC, evaluation)
# ============================================================

def s_combo(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    sx = torch.tanh((x - (0.45 + 0.03 * torch.sin(2.0 * PI * y))) / DELTA)
    sy = torch.tanh((y - (0.58 + 0.02 * torch.cos(2.0 * PI * x))) / DELTA)
    return 0.55 * sx + 0.45 * sy + 0.08 * torch.sin(2.0 * PI * x + 0.8 * PI * y)


def psi_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    A = 0.10
    B = 0.14
    C = 0.06
    s = s_combo(x, y)
    return (
        A * torch.sin(PI * x) * torch.sin(PI * y)
        + B * s * torch.sin(PI * y)
        + C * torch.sin(2.0 * PI * x) * torch.sin(2.0 * PI * y)
        + 0.02 * torch.cos(3.0 * PI * x) * torch.sin(2.0 * PI * y)
    )


def p_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s = s_combo(x, y)
    return 1.0 + 0.21 * s + 0.05 * torch.cos(2.0 * PI * y) + 0.012 * torch.sin(2.0 * PI * x + 0.8 * PI * y)


def phi_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s = s_combo(x, y)
    return (
        0.34 * torch.sin(2.0 * PI * x) * torch.sin(2.0 * PI * y)
        + 0.15 * s * torch.cos(PI * y)
        + 0.02 * torch.sin(3.0 * PI * x) * torch.sin(PI * y)
        + 0.012 * torch.cos(2.0 * PI * x + 0.8 * PI * y)
    )


def T_star(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    s = s_combo(x, y)
    return (
        300.0
        + 12.0 * s
        + 7.0 * torch.sin(2.0 * PI * x) * torch.cos(2.0 * PI * y)
        + 1.2 * torch.cos(PI * x) * torch.sin(3.0 * PI * y)
        + 0.9 * torch.sin(2.0 * PI * x + 0.8 * PI * y)
    )


def truth_u_v_from_psi(x: torch.Tensor, y: torch.Tensor, create_graph: bool):
    psi = psi_star(x, y)
    psi_x = grad(psi, x, create_graph=create_graph)
    psi_y = grad(psi, y, create_graph=create_graph)
    u = psi_y
    v = -psi_x
    return u, v, psi


def truth_rho_and_sources(x_in: torch.Tensor, y_in: torch.Tensor, create_graph: bool):
    x = x_in
    y = y_in

    u, v, _ = truth_u_v_from_psi(x, y, create_graph=True)
    p = p_star(x, y)
    phi = phi_star(x, y)
    T = T_star(x, y)

    # phi derivatives and screened-Poisson rho_e
    phi_x = grad(phi, x, create_graph=True)
    phi_y = grad(phi, y, create_graph=True)
    phi_xx = grad(phi_x, x, create_graph=True)
    phi_yy = grad(phi_y, y, create_graph=True)
    rho_e = -EPS * (phi_xx + phi_yy) + (KAPPA_D ** 2) * phi

    # velocity derivatives
    u_x = grad(u, x, create_graph=True)
    u_y = grad(u, y, create_graph=True)
    u_xx = grad(u_x, x, create_graph=True)
    u_yy = grad(u_y, y, create_graph=True)

    v_x = grad(v, x, create_graph=True)
    v_y = grad(v, y, create_graph=True)
    v_xx = grad(v_x, x, create_graph=True)
    v_yy = grad(v_y, y, create_graph=True)

    # pressure derivatives up to 3rd (for Laplacian on gradient)
    p_x = grad(p, x, create_graph=True)
    p_y = grad(p, y, create_graph=True)

    p_xx = grad(p_x, x, create_graph=True)
    p_xy = grad(p_x, y, create_graph=True)
    p_yy = grad(p_y, y, create_graph=True)
    p_yx = grad(p_y, x, create_graph=True)

    p_xxx = grad(p_xx, x, create_graph=True)
    p_xyy = grad(p_xy, y, create_graph=True)
    p_yxx = grad(p_yx, x, create_graph=True)
    p_yyy = grad(p_yy, y, create_graph=True)

    lap_px = p_xxx + p_xyy
    lap_py = p_yxx + p_yyy

    # temperature derivatives
    T_x = grad(T, x, create_graph=True)
    T_y = grad(T, y, create_graph=True)
    T_xx = grad(T_x, x, create_graph=True)
    T_yy = grad(T_y, y, create_graph=True)

    lap_u = u_xx + u_yy
    lap_v = v_xx + v_yy

    f_u = u * u_x + v * u_y + p_x - NU0 * lap_u + ALPHA_E * rho_e * phi_x + C_P * lap_px
    f_v = u * v_x + v * v_y + p_y - NU0 * lap_v + ALPHA_E * rho_e * phi_y + C_P * lap_py
    f_T = u * T_x + v * T_y - KAPPA * (T_xx + T_yy) - SIGMA_J * (phi_x * phi_x + phi_y * phi_y)

    if not create_graph:
        rho_e = rho_e.detach()
        f_u = f_u.detach()
        f_v = f_v.detach()
        f_T = f_T.detach()

    return rho_e, f_u, f_v, f_T


# ============================================================
# Two-expert gated model (NO truth inside forward)
# Outputs tilde variables: psi_tilde, p_tilde, phi_tilde, T_tilde
# ============================================================

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int, depth: int, act: str):
        super().__init__()
        if act == "tanh":
            self.act = nn.Tanh()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError("act must be 'tanh' or 'silu'")

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(in_dim, width))
        for _ in range(depth - 1):
            self.layers.append(nn.Linear(width, width))
        self.out = nn.Linear(width, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)  # final bias=0 => tilde ~ 0 at init

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = self.act(layer(h))
        return self.out(h)


class TwoExpertGated(nn.Module):
    def __init__(self, width: int, depth: int, act: str):
        super().__init__()
        in_dim = 3
        out_dim = 4
        self.left = MLP(in_dim, out_dim, width, depth, act)
        self.right = MLP(in_dim, out_dim, width, depth, act)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        xi = feature_xi(x, y)
        feats = torch.cat([x, y, xi], dim=1)
        outL = self.left(feats)
        outR = self.right(feats)
        g = gate_g(x, y)
        out = (1.0 - g) * outL + g * outR
        w_int = sech2(xi)
        return out, outL, outR, w_int


# ============================================================
# Prediction + derivatives for residuals
# ============================================================

def predict_fields_and_derivs(model: TwoExpertGated,
                              x_in: torch.Tensor,
                              y_in: torch.Tensor,
                              need_param_grads: bool):
    x = x_in
    y = y_in
    create_graph_final = need_param_grads

    out, outL, outR, w_int = model(x, y)
    psi_tilde = out[:, 0:1]
    p_tilde = out[:, 1:2]
    phi_tilde = out[:, 2:3]
    T_tilde = out[:, 3:4]

    psi = PSI0 * psi_tilde
    p = 1.0 + P0 * p_tilde
    phi = PHI0 * phi_tilde
    T = T0 + T1 * T_tilde

    # psi derivatives up to 3rd
    psi_x = grad(psi, x, create_graph=True)
    psi_y = grad(psi, y, create_graph=True)

    psi_xx = grad(psi_x, x, create_graph=True)
    psi_xy = grad(psi_x, y, create_graph=True)
    psi_yy = grad(psi_y, y, create_graph=True)

    psi_xxx = grad(psi_xx, x, create_graph=create_graph_final)
    psi_xxy = grad(psi_xx, y, create_graph=create_graph_final)
    psi_xyy = grad(psi_xy, y, create_graph=create_graph_final)
    psi_yyy = grad(psi_yy, y, create_graph=create_graph_final)

    u = psi_y
    v = -psi_x

    u_x = psi_xy
    u_y = psi_yy
    u_xx = psi_xxy
    u_yy = psi_yyy

    v_x = -psi_xx
    v_y = -psi_xy
    v_xx = -psi_xxx
    v_yy = -psi_xyy

    # p derivatives up to 3rd for Laplacian(grad p)
    p_x = grad(p, x, create_graph=True)
    p_y = grad(p, y, create_graph=True)

    p_xx = grad(p_x, x, create_graph=True)
    p_xy = grad(p_x, y, create_graph=True)
    p_yy = grad(p_y, y, create_graph=True)
    p_yx = grad(p_y, x, create_graph=True)

    p_xxx = grad(p_xx, x, create_graph=create_graph_final)
    p_xyy = grad(p_xy, y, create_graph=create_graph_final)
    p_yxx = grad(p_yx, x, create_graph=create_graph_final)
    p_yyy = grad(p_yy, y, create_graph=create_graph_final)

    lap_px = p_xxx + p_xyy
    lap_py = p_yxx + p_yyy

    # phi derivatives (2nd)
    phi_x = grad(phi, x, create_graph=True)
    phi_y = grad(phi, y, create_graph=True)
    phi_xx = grad(phi_x, x, create_graph=create_graph_final)
    phi_yy = grad(phi_y, y, create_graph=create_graph_final)

    # T derivatives (2nd)
    T_x = grad(T, x, create_graph=True)
    T_y = grad(T, y, create_graph=True)
    T_xx = grad(T_x, x, create_graph=create_graph_final)
    T_yy = grad(T_y, y, create_graph=create_graph_final)

    return {
        "psi_tilde": psi_tilde,
        "p_tilde": p_tilde,
        "phi_tilde": phi_tilde,
        "T_tilde": T_tilde,
        "u": u,
        "v": v,
        "p": p,
        "phi": phi,
        "T": T,
        "u_x": u_x,
        "u_y": u_y,
        "u_xx": u_xx,
        "u_yy": u_yy,
        "v_x": v_x,
        "v_y": v_y,
        "v_xx": v_xx,
        "v_yy": v_yy,
        "p_x": p_x,
        "p_y": p_y,
        "lap_px": lap_px,
        "lap_py": lap_py,
        "phi_x": phi_x,
        "phi_y": phi_y,
        "phi_xx": phi_xx,
        "phi_yy": phi_yy,
        "T_x": T_x,
        "T_y": T_y,
        "T_xx": T_xx,
        "T_yy": T_yy,
        "outL": outL,
        "outR": outR,
        "w_int": w_int,
        "_x": x,
        "_y": y,
    }


def residuals_normalized(pred: dict,
                         rho_e: torch.Tensor,
                         f_u: torch.Tensor,
                         f_v: torch.Tensor,
                         f_T: torch.Tensor,
                         scales: dict):
    u = pred["u"]
    v = pred["v"]

    lap_u = pred["u_xx"] + pred["u_yy"]
    lap_v = pred["v_xx"] + pred["v_yy"]

    r_u = u * pred["u_x"] + v * pred["u_y"] + pred["p_x"] - NU0 * lap_u + ALPHA_E * rho_e * pred["phi_x"] + C_P * pred["lap_px"] - f_u
    r_v = u * pred["v_x"] + v * pred["v_y"] + pred["p_y"] - NU0 * lap_v + ALPHA_E * rho_e * pred["phi_y"] + C_P * pred["lap_py"] - f_v
    r_phi = -EPS * (pred["phi_xx"] + pred["phi_yy"]) + (KAPPA_D ** 2) * pred["phi"] - rho_e
    r_T = u * pred["T_x"] + v * pred["T_y"] - KAPPA * (pred["T_xx"] + pred["T_yy"]) - SIGMA_J * (pred["phi_x"] * pred["phi_x"] + pred["phi_y"] * pred["phi_y"]) - f_T

    rhat_u = r_u / scales["Su"]
    rhat_v = r_v / scales["Sv"]
    rhat_phi = r_phi / scales["Sphi"]
    rhat_T = r_T / scales["ST"]
    return rhat_u, rhat_v, rhat_phi, rhat_T


# ============================================================
# Sampling utilities (robust shapes, no AxisError)
# ============================================================

def sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    xy = rng.random((n, 2))
    assert xy.ndim == 2 and xy.shape[1] == 2
    return xy


def sample_band_stripes(rng: np.random.Generator, n: int) -> np.ndarray:
    n1 = n // 2
    n2 = n - n1

    y1 = rng.random((n1, 1))
    x_center = 0.45 + 0.03 * np.sin(2.0 * PI * y1)
    dx = (2.0 * rng.random((n1, 1)) - 1.0) * BAND_W
    x1 = np.clip(x_center + dx, 0.0, 1.0)
    xy1 = np.concatenate([x1, y1], axis=1)

    x2 = rng.random((n2, 1))
    y_center = 0.58 + 0.02 * np.cos(2.0 * PI * x2)
    dy = (2.0 * rng.random((n2, 1)) - 1.0) * BAND_W
    y2 = np.clip(y_center + dy, 0.0, 1.0)
    xy2 = np.concatenate([x2, y2], axis=1)

    xy = np.concatenate([xy1, xy2], axis=0)
    rng.shuffle(xy)
    assert xy.shape == (n, 2)
    return xy


def build_initial_interior_pool(rng: np.random.Generator, nf_init: int) -> np.ndarray:
    n_band = int(round(BAND_RATIO * nf_init))
    n_uni = int(nf_init - n_band)
    xy_band = sample_band_stripes(rng, n_band)
    xy_uni = sample_uniform(rng, n_uni)
    assert xy_band.ndim == 2 and xy_band.shape[1] == 2
    assert xy_uni.ndim == 2 and xy_uni.shape[1] == 2
    xy = np.concatenate([xy_band, xy_uni], axis=0)
    rng.shuffle(xy)
    assert xy.shape == (nf_init, 2)
    return xy


def build_boundary_points(n_edge: int) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, n_edge)
    ys = np.linspace(0.0, 1.0, n_edge)
    bottom = np.stack([xs, np.zeros_like(xs)], axis=1)
    top = np.stack([xs, np.ones_like(xs)], axis=1)
    left = np.stack([np.zeros_like(ys), ys], axis=1)
    right = np.stack([np.ones_like(ys), ys], axis=1)
    xy = np.concatenate([bottom, top, left, right], axis=0)
    assert xy.ndim == 2 and xy.shape[1] == 2
    return xy


def build_grid(nx: int, ny: int) -> tuple:
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")
    xy = np.stack([X.ravel(), Y.ravel()], axis=1)
    assert xy.ndim == 2 and xy.shape[1] == 2
    return xy, X, Y


# ============================================================
# IO: plots and triplet text
# ============================================================

def save_triplet_txt(path_txt: str, X: np.ndarray, Y: np.ndarray, V: np.ndarray) -> None:
    data = np.column_stack([X.ravel(), Y.ravel(), V.ravel()])
    np.savetxt(path_txt, data, fmt="%.12e")


def plot_field(path_png: str, X: np.ndarray, Y: np.ndarray, V: np.ndarray, vmin: float, vmax: float, title: str) -> None:
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


def plot_curve(path_png: str, xs: np.ndarray, ys_list: list, labels: list, ylog: bool, xlabel: str, ylabel: str, title: str) -> None:
    fig = plt.figure(figsize=(7.0, 4.5), dpi=150)
    ax = fig.add_subplot(111)
    for y, lab in zip(ys_list, labels):
        ax.plot(xs, y, label=lab)
    if ylog:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=300)
    plt.close(fig)


# ============================================================
# Hard buffer
# ============================================================

class HardBuffer:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.xy = None
        self.rhs = None
        self.score = None

    def size(self) -> int:
        if self.xy is None:
            return 0
        return int(self.xy.shape[0])

    def add(self, xy_new: torch.Tensor, rhs_new: torch.Tensor, score_new: torch.Tensor):
        if xy_new.numel() == 0:
            return
        if self.xy is None:
            self.xy = xy_new.clone()
            self.rhs = rhs_new.clone()
            self.score = score_new.clone()
        else:
            self.xy = torch.cat([self.xy, xy_new], dim=0)
            self.rhs = torch.cat([self.rhs, rhs_new], dim=0)
            self.score = torch.cat([self.score, score_new], dim=0)

        # approximate dedup by rounding to 1e-5
        xy_round = torch.round(self.xy * 100000.0)
        key = xy_round[:, 0] * 1000000.0 + xy_round[:, 1]
        idx = torch.argsort(key)
        key_s = key[idx]
        score_s = self.score[idx].squeeze(1)
        xy_s = self.xy[idx]
        rhs_s = self.rhs[idx]

        keep = []
        start = 0
        n = int(key_s.shape[0])
        while start < n:
            k = key_s[start]
            end = start + 1
            while end < n and key_s[end] == k:
                end += 1
            seg = score_s[start:end]
            imax = torch.argmax(seg).item()
            keep.append(start + imax)
            start = end
        keep = torch.tensor(keep, dtype=torch.long)

        self.xy = xy_s[keep]
        self.rhs = rhs_s[keep]
        self.score = score_s[keep].unsqueeze(1)

        if self.xy.shape[0] > self.capacity:
            _, topi = torch.topk(self.score.squeeze(1), k=self.capacity, largest=True, sorted=False)
            self.xy = self.xy[topi]
            self.rhs = self.rhs[topi]
            self.score = self.score[topi]

    def sample(self, n: int):
        m = self.size()
        if m <= 0:
            return None, None
        n_take = min(int(n), m)
        idx = torch.randint(low=0, high=m, size=(n_take,), dtype=torch.long)
        return self.xy[idx], self.rhs[idx]


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--device", type=str, default=DEVICE_DEFAULT)
    args = parser.parse_args()

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    root = "case10"
    figs_dir = os.path.join(root, "figs")
    data_dir = os.path.join(root, "data")
    logs_dir = os.path.join(root, "logs")
    ensure_dir(figs_dir)
    ensure_dir(data_dir)
    ensure_dir(logs_dir)

    interior_pool = build_initial_interior_pool(rng, NF_INIT)
    Nf_current = int(interior_pool.shape[0])

    boundary_xy = build_boundary_points(N_EDGE)
    Nb = int(boundary_xy.shape[0])

    cert_xy_np, _, _ = build_grid(NX_CERT, NY_CERT)
    Nc = int(cert_xy_np.shape[0])

    eval_xy_np, eval_X_np, eval_Y_np = build_grid(NX_EVAL, NY_EVAL)
    Ne = int(eval_xy_np.shape[0])

    # RHS on cert grid
    t0_rhs = now_s()
    cert_rhs = np.zeros((Nc, 4), dtype=np.float64)
    for i, j in chunked_range(Nc, CERT_BATCH):
        xy = torch.from_numpy(cert_xy_np[i:j]).to(device=device)
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        rho_e, f_u, f_v, f_T = truth_rho_and_sources(x, y, create_graph=False)
        cert_rhs[i:j, :] = torch.cat([rho_e, f_u, f_v, f_T], dim=1).detach().cpu().numpy()
        del xy, x, y, rho_e, f_u, f_v, f_T
    rhs_time_s = now_s() - t0_rhs

    # RHS scales
    t0_scale = now_s()
    xy_scale = rng.random((2000, 2))
    xy_scale_t = torch.from_numpy(xy_scale).to(device=device)
    xs = xy_scale_t[:, 0:1].clone().detach().requires_grad_(True)
    ys = xy_scale_t[:, 1:2].clone().detach().requires_grad_(True)
    rho_s, fu_s, fv_s, fT_s = truth_rho_and_sources(xs, ys, create_graph=False)
    Su = float(rms(fu_s).item() + 1e-12)
    Sv = float(rms(fv_s).item() + 1e-12)
    Sphi = float(rms(rho_s).item() + 1e-12)
    ST = float(rms(fT_s).item() + 1e-12)
    scales = {"Su": Su, "Sv": Sv, "Sphi": Sphi, "ST": ST}
    scale_time_s = now_s() - t0_scale
    del xy_scale_t, xs, ys, rho_s, fu_s, fv_s, fT_s

    # boundary truth
    t0_bc = now_s()
    bxy_t = torch.from_numpy(boundary_xy).to(device=device)
    xb = bxy_t[:, 0:1].clone().detach().requires_grad_(True)
    yb = bxy_t[:, 1:2].clone().detach().requires_grad_(True)
    u_b, v_b, psi_b = truth_u_v_from_psi(xb, yb, create_graph=False)
    p_b = p_star(xb, yb).detach()
    phi_b = phi_star(xb, yb).detach()
    T_b = T_star(xb, yb).detach()

    psi_tilde_bc = (psi_b / PSI0).detach()
    p_tilde_bc = ((p_b - 1.0) / P0).detach()
    phi_tilde_bc = (phi_b / PHI0).detach()
    T_tilde_bc = ((T_b - T0) / T1).detach()
    u_bc = u_b.detach()
    v_bc = v_b.detach()
    bc_time_s = now_s() - t0_bc
    del bxy_t, xb, yb, u_b, v_b, psi_b, p_b, phi_b, T_b

    # Model/opt
    model = TwoExpertGated(width=WIDTH, depth=DEPTH, act=ACTIVATION).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)

    def lr_at_step(step: int) -> float:
        lr0 = LR_INIT
        lr1 = LR_INIT * LR_FINAL_RATIO
        t = float(step) / float(max(1, MAX_STEPS))
        return lr1 + 0.5 * (lr0 - lr1) * (1.0 + np.cos(np.pi * t))

    buffer = HardBuffer(capacity=BUFFER_SIZE)

    loss_hist_path = os.path.join(logs_dir, "loss_history.txt")
    with open(loss_hist_path, "w", encoding="utf-8") as f:
        f.write("step lr L_total L_phys L_bc L_cert L_int cert_max Nf_current buffer_size\n")

    cert_max_last = float("inf")
    cert_history = []
    stagnation_count = 0

    xb_all = torch.from_numpy(boundary_xy[:, 0:1]).to(device=device)
    yb_all = torch.from_numpy(boundary_xy[:, 1:2]).to(device=device)

    def rhs_on_the_fly(xy: torch.Tensor) -> torch.Tensor:
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        rho_e, f_u, f_v, f_T = truth_rho_and_sources(x, y, create_graph=False)
        return torch.cat([rho_e, f_u, f_v, f_T], dim=1).detach()

    def compute_losses(xy_f_t: torch.Tensor, rhs_f_t: torch.Tensor, idx_b: torch.Tensor):
        x = xy_f_t[:, 0:1].clone().detach().requires_grad_(True)
        y = xy_f_t[:, 1:2].clone().detach().requires_grad_(True)
        pred = predict_fields_and_derivs(model, x, y, need_param_grads=True)

        rho_e = rhs_f_t[:, 0:1]
        f_u = rhs_f_t[:, 1:2]
        f_v = rhs_f_t[:, 2:3]
        f_T = rhs_f_t[:, 3:4]

        rhat_u, rhat_v, rhat_phi, rhat_T = residuals_normalized(pred, rho_e, f_u, f_v, f_T, scales)
        L_phys = torch.mean(rhat_u * rhat_u + rhat_v * rhat_v + rhat_phi * rhat_phi + rhat_T * rhat_T)

        outL = pred["outL"]
        outR = pred["outR"]
        w_int_local = pred["w_int"]
        L_int = torch.mean(w_int_local * (outL - outR).pow(2).sum(dim=1, keepdim=True))

        rhat_max = torch.max(torch.max(torch.abs(rhat_u), torch.abs(rhat_v)),
                             torch.max(torch.abs(rhat_phi), torch.abs(rhat_T)))
        z = rhat_max.pow(2)
        zmax = torch.max(z).detach()
        L_cert = (1.0 / BETA_CERT) * (torch.log(torch.mean(torch.exp(BETA_CERT * (z - zmax)))) + BETA_CERT * zmax)

        xb_sel = xb_all[idx_b].clone().detach().requires_grad_(True)
        yb_sel = yb_all[idx_b].clone().detach().requires_grad_(True)
        pred_b = predict_fields_and_derivs(model, xb_sel, yb_sel, need_param_grads=True)

        mse = nn.MSELoss(reduction="mean")
        L_bc = (
            mse(pred_b["psi_tilde"], psi_tilde_bc[idx_b])
            + mse(pred_b["u"], u_bc[idx_b])
            + mse(pred_b["v"], v_bc[idx_b])
            + mse(pred_b["p_tilde"], p_tilde_bc[idx_b])
            + mse(pred_b["phi_tilde"], phi_tilde_bc[idx_b])
            + mse(pred_b["T_tilde"], T_tilde_bc[idx_b])
        )

        L_total = L_phys + W_BC * L_bc + W_CERT * L_cert + W_INT * L_int
        return L_total, L_phys, L_bc, L_cert, L_int

    def certificate_scan_and_adapt(step: int):
        nonlocal cert_max_last, interior_pool, Nf_current, stagnation_count

        cert_u = 0.0
        cert_v = 0.0
        cert_phi = 0.0
        cert_T = 0.0

        top_u_val = torch.full((0,), -1.0, dtype=torch.float64)
        top_u_xy = torch.empty((0, 2), dtype=torch.float64)
        top_u_rhs = torch.empty((0, 4), dtype=torch.float64)

        top_v_val = torch.full((0,), -1.0, dtype=torch.float64)
        top_v_xy = torch.empty((0, 2), dtype=torch.float64)
        top_v_rhs = torch.empty((0, 4), dtype=torch.float64)

        top_phi_val = torch.full((0,), -1.0, dtype=torch.float64)
        top_phi_xy = torch.empty((0, 2), dtype=torch.float64)
        top_phi_rhs = torch.empty((0, 4), dtype=torch.float64)

        top_T_val = torch.full((0,), -1.0, dtype=torch.float64)
        top_T_xy = torch.empty((0, 2), dtype=torch.float64)
        top_T_rhs = torch.empty((0, 4), dtype=torch.float64)

        top_max_val = torch.full((0,), -1.0, dtype=torch.float64)
        top_max_xy = torch.empty((0, 2), dtype=torch.float64)
        top_max_rhs = torch.empty((0, 4), dtype=torch.float64)

        def update_topk(cur_val, cur_xy, cur_rhs, new_val, new_xy, new_rhs, K):
            if new_val.numel() == 0:
                return cur_val, cur_xy, cur_rhs
            val = torch.cat([cur_val, new_val], dim=0)
            xy = torch.cat([cur_xy, new_xy], dim=0)
            rhs = torch.cat([cur_rhs, new_rhs], dim=0)
            if val.shape[0] <= K:
                return val, xy, rhs
            tv, ti = torch.topk(val, k=K, largest=True, sorted=False)
            return tv, xy[ti], rhs[ti]

        model.eval()
        for i, j in chunked_range(Nc, CERT_BATCH):
            xy_np = cert_xy_np[i:j]
            rhs_np = cert_rhs[i:j]
            xy = torch.from_numpy(xy_np).to(device=device)
            rhs = torch.from_numpy(rhs_np).to(device=device)

            x = xy[:, 0:1].clone().detach().requires_grad_(True)
            y = xy[:, 1:2].clone().detach().requires_grad_(True)
            pred = predict_fields_and_derivs(model, x, y, need_param_grads=False)

            rho_e = rhs[:, 0:1]
            f_u = rhs[:, 1:2]
            f_v = rhs[:, 2:3]
            f_T = rhs[:, 3:4]
            rhat_u, rhat_v, rhat_phi, rhat_T = residuals_normalized(pred, rho_e, f_u, f_v, f_T, scales)

            au = torch.abs(rhat_u).detach()
            av = torch.abs(rhat_v).detach()
            ap = torch.abs(rhat_phi).detach()
            aT = torch.abs(rhat_T).detach()
            rmax = torch.max(torch.max(au, av), torch.max(ap, aT))

            cert_u = max(cert_u, float(torch.max(au).item()))
            cert_v = max(cert_v, float(torch.max(av).item()))
            cert_phi = max(cert_phi, float(torch.max(ap).item()))
            cert_T = max(cert_T, float(torch.max(aT).item()))

            xy_cpu = torch.from_numpy(xy_np).to(dtype=torch.float64)
            rhs_cpu = torch.from_numpy(rhs_np).to(dtype=torch.float64)

            ku = min(TOPK_U, au.shape[0])
            tv, ti = torch.topk(au.squeeze(1), k=ku, largest=True, sorted=False)
            top_u_val, top_u_xy, top_u_rhs = update_topk(top_u_val, top_u_xy, top_u_rhs, tv.cpu(), xy_cpu[ti.cpu()], rhs_cpu[ti.cpu()], TOPK_U)

            kv = min(TOPK_V, av.shape[0])
            tv, ti = torch.topk(av.squeeze(1), k=kv, largest=True, sorted=False)
            top_v_val, top_v_xy, top_v_rhs = update_topk(top_v_val, top_v_xy, top_v_rhs, tv.cpu(), xy_cpu[ti.cpu()], rhs_cpu[ti.cpu()], TOPK_V)

            kp = min(TOPK_PHI, ap.shape[0])
            tv, ti = torch.topk(ap.squeeze(1), k=kp, largest=True, sorted=False)
            top_phi_val, top_phi_xy, top_phi_rhs = update_topk(top_phi_val, top_phi_xy, top_phi_rhs, tv.cpu(), xy_cpu[ti.cpu()], rhs_cpu[ti.cpu()], TOPK_PHI)

            kt = min(TOPK_T, aT.shape[0])
            tv, ti = torch.topk(aT.squeeze(1), k=kt, largest=True, sorted=False)
            top_T_val, top_T_xy, top_T_rhs = update_topk(top_T_val, top_T_xy, top_T_rhs, tv.cpu(), xy_cpu[ti.cpu()], rhs_cpu[ti.cpu()], TOPK_T)

            km = min(TOPK_MAX, rmax.shape[0])
            tv, ti = torch.topk(rmax.squeeze(1), k=km, largest=True, sorted=False)
            top_max_val, top_max_xy, top_max_rhs = update_topk(top_max_val, top_max_xy, top_max_rhs, tv.cpu(), xy_cpu[ti.cpu()], rhs_cpu[ti.cpu()], TOPK_MAX)

            del xy, rhs, x, y, pred, rhat_u, rhat_v, rhat_phi, rhat_T, au, av, ap, aT, rmax

        model.train()

        cert_max = max(cert_u, cert_v, cert_phi, cert_T)

        if len(cert_history) > 0:
            prev = cert_history[-1]
            rel_drop = (prev - cert_max) / max(prev, 1e-12)
            if rel_drop < CERT_MIN_REL_DROP:
                stagnation_count += 1
            else:
                stagnation_count = 0
            if stagnation_count >= CERT_STAGNATION_WINDOW:
                for pg in optimizer.param_groups:
                    pg["lr"] *= LR_HALVE_FACTOR
                stagnation_count = 0

        cert_history.append(cert_max)
        cert_max_last = cert_max

        xy_cat = torch.cat([top_u_xy, top_v_xy, top_phi_xy, top_T_xy, top_max_xy], dim=0)
        rhs_cat = torch.cat([top_u_rhs, top_v_rhs, top_phi_rhs, top_T_rhs, top_max_rhs], dim=0)
        score_cat = torch.cat([top_u_val, top_v_val, top_phi_val, top_T_val, top_max_val], dim=0).unsqueeze(1)
        buffer.add(xy_cat, rhs_cat, score_cat)

        add_max = top_max_xy.numpy()
        add_band = sample_band_stripes(rng, ADD_BAND_PER_SCAN)
        interior_pool = np.concatenate([interior_pool, add_max, add_band], axis=0)
        Nf_current = int(interior_pool.shape[0])

        return cert_u, cert_v, cert_phi, cert_T, cert_max

    t0_train = now_s()
    cert_u0, cert_v0, cert_phi0, cert_T0, cert0 = certificate_scan_and_adapt(step=0)

    print(f"[init] dtype={torch.get_default_dtype()} device={device} max_steps={MAX_STEPS} cert_interval={CERT_INTERVAL}")
    print(f"[init] Nf_init={NF_INIT} Nb={Nb} Nc_cert={Nc} Nx_eval={NX_EVAL} Ny_eval={NY_EVAL}")
    print(f"[init] rhs_precompute_s={rhs_time_s:.3f} scale_s={scale_time_s:.3f} bc_prep_s={bc_time_s:.3f}")
    print(f"[scale] Su={Su:.6e} Sv={Sv:.6e} Sphi={Sphi:.6e} ST={ST:.6e}")
    print(f"[cert-init] step=0 cert_u={cert_u0:.3e} cert_v={cert_v0:.3e} cert_phi={cert_phi0:.3e} cert_T={cert_T0:.3e} cert_max={cert0:.3e} Nf={Nf_current} buffer={buffer.size()}")

    model.train()
    for step in range(1, MAX_STEPS + 1):
        lr_now = float(lr_at_step(step))
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        if step % CERT_INTERVAL == 0:
            certificate_scan_and_adapt(step=step)
        cert_now = cert_max_last

        n_buf = int(round(BUF_RATIO_IN_BATCH * BATCH_F))
        n_reg = int(BATCH_F - n_buf)

        xy_buf_cpu, rhs_buf_cpu = buffer.sample(n_buf)
        if xy_buf_cpu is None or rhs_buf_cpu is None or int(xy_buf_cpu.shape[0]) < n_buf:
            n_buf_eff = 0 if xy_buf_cpu is None else int(xy_buf_cpu.shape[0])
            n_reg = int(BATCH_F - n_buf_eff)
        else:
            n_buf_eff = n_buf

        idx_reg = rng.integers(low=0, high=Nf_current, size=(n_reg,))
        xy_reg_np = interior_pool[idx_reg, :]
        xy_reg_t = torch.from_numpy(xy_reg_np).to(device=device)
        rhs_reg_t = rhs_on_the_fly(xy_reg_t)

        if n_buf_eff > 0:
            xy_buf_t = xy_buf_cpu.to(device=device)
            rhs_buf_t = rhs_buf_cpu.to(device=device)
            xy_f_t = torch.cat([xy_buf_t, xy_reg_t], dim=0)
            rhs_f_t = torch.cat([rhs_buf_t, rhs_reg_t], dim=0)
        else:
            xy_f_t = xy_reg_t
            rhs_f_t = rhs_reg_t

        idx_b = torch.randint(low=0, high=Nb, size=(BATCH_B,), device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        L_total, L_phys, L_bc, L_cert, L_int = compute_losses(xy_f_t, rhs_f_t, idx_b)
        L_total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        with open(loss_hist_path, "a", encoding="utf-8") as f:
            f.write(f"{step} {lr_now:.12e} {L_total.item():.12e} {L_phys.item():.12e} {L_bc.item():.12e} {L_cert.item():.12e} {L_int.item():.12e} {float(cert_now):.12e} {Nf_current} {buffer.size()}\n")

        bar = ascii_bar(step, MAX_STEPS, width=24)
        print(f"{bar} step={step}/{MAX_STEPS} "
              f"lr={lr_now:.2e} "
              f"L={L_total.item():.3e} Lp={L_phys.item():.3e} Lbc={L_bc.item():.3e} Lcert={L_cert.item():.3e} Lint={L_int.item():.3e} "
              f"cert={float(cert_now):.3e} Nf={Nf_current} buf={buffer.size()}")

    train_time_s = now_s() - t0_train

    # evaluation
    t0_eval = now_s()
    u_pred = np.zeros((Ne,), dtype=np.float64)
    v_pred = np.zeros((Ne,), dtype=np.float64)
    p_pred = np.zeros((Ne,), dtype=np.float64)
    phi_pred = np.zeros((Ne,), dtype=np.float64)
    T_pred = np.zeros((Ne,), dtype=np.float64)

    model.eval()
    for i, j in chunked_range(Ne, EVAL_BATCH):
        xy = torch.from_numpy(eval_xy_np[i:j]).to(device=device)
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        pred = predict_fields_and_derivs(model, x, y, need_param_grads=False)
        u_pred[i:j] = pred["u"].detach().cpu().numpy().ravel()
        v_pred[i:j] = pred["v"].detach().cpu().numpy().ravel()
        p_pred[i:j] = pred["p"].detach().cpu().numpy().ravel()
        phi_pred[i:j] = pred["phi"].detach().cpu().numpy().ravel()
        T_pred[i:j] = pred["T"].detach().cpu().numpy().ravel()
        del xy, x, y, pred
    model.train()

    u_true = np.zeros((Ne,), dtype=np.float64)
    v_true = np.zeros((Ne,), dtype=np.float64)
    p_true = np.zeros((Ne,), dtype=np.float64)
    phi_true = np.zeros((Ne,), dtype=np.float64)
    T_true = np.zeros((Ne,), dtype=np.float64)

    for i, j in chunked_range(Ne, EVAL_BATCH):
        xy = torch.from_numpy(eval_xy_np[i:j]).to(device=device)
        x = xy[:, 0:1].clone().detach().requires_grad_(True)
        y = xy[:, 1:2].clone().detach().requires_grad_(True)
        u_t, v_t, _ = truth_u_v_from_psi(x, y, create_graph=False)
        p_t = p_star(x, y).detach()
        phi_t = phi_star(x, y).detach()
        T_t = T_star(x, y).detach()
        u_true[i:j] = u_t.detach().cpu().numpy().ravel()
        v_true[i:j] = v_t.detach().cpu().numpy().ravel()
        p_true[i:j] = p_t.cpu().numpy().ravel()
        phi_true[i:j] = phi_t.cpu().numpy().ravel()
        T_true[i:j] = T_t.cpu().numpy().ravel()
        del xy, x, y, u_t, v_t, p_t, phi_t, T_t

    eval_time_s = now_s() - t0_eval
    total_time_s = train_time_s + eval_time_s

    U_pred = u_pred.reshape((NY_EVAL, NX_EVAL))
    V_pred = v_pred.reshape((NY_EVAL, NX_EVAL))
    P_pred = p_pred.reshape((NY_EVAL, NX_EVAL))
    PHI_pred = phi_pred.reshape((NY_EVAL, NX_EVAL))
    TT_pred = T_pred.reshape((NY_EVAL, NX_EVAL))

    U_true = u_true.reshape((NY_EVAL, NX_EVAL))
    V_true = v_true.reshape((NY_EVAL, NX_EVAL))
    P_true = p_true.reshape((NY_EVAL, NX_EVAL))
    PHI_true = phi_true.reshape((NY_EVAL, NX_EVAL))
    TT_true = T_true.reshape((NY_EVAL, NX_EVAL))

    U_err = np.abs(U_pred - U_true)
    V_err = np.abs(V_pred - V_true)
    P_err = np.abs(P_pred - P_true)
    PHI_err = np.abs(PHI_pred - PHI_true)
    TT_err = np.abs(TT_pred - TT_true)

    def dump_all(prefix: str, true_grid: np.ndarray, pred_grid: np.ndarray, err_grid: np.ndarray):
        save_triplet_txt(os.path.join(data_dir, f"{prefix}_true.txt"), eval_X_np, eval_Y_np, true_grid)
        save_triplet_txt(os.path.join(data_dir, f"{prefix}_pred.txt"), eval_X_np, eval_Y_np, pred_grid)
        save_triplet_txt(os.path.join(data_dir, f"{prefix}_maxerror.txt"), eval_X_np, eval_Y_np, err_grid)

        vmin = float(np.min(true_grid))
        vmax = float(np.max(true_grid))
        emax = float(np.max(err_grid))

        plot_field(os.path.join(figs_dir, f"{prefix}_true.png"), eval_X_np, eval_Y_np, true_grid, vmin, vmax, f"{prefix} true")
        plot_field(os.path.join(figs_dir, f"{prefix}_pred.png"), eval_X_np, eval_Y_np, pred_grid, vmin, vmax, f"{prefix} pred")
        plot_field(os.path.join(figs_dir, f"{prefix}_maxerror.png"), eval_X_np, eval_Y_np, err_grid, 0.0, emax, f"{prefix} maxerror")

    dump_all("u", U_true, U_pred, U_err)
    dump_all("v", V_true, V_pred, V_err)
    dump_all("p", P_true, P_pred, P_err)
    dump_all("phi", PHI_true, PHI_pred, PHI_err)
    dump_all("t", TT_true, TT_pred, TT_err)

    def metrics(true_grid: np.ndarray, pred_grid: np.ndarray):
        e = pred_grid - true_grid
        mse = float(np.mean(e * e))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(np.abs(e)))
        l2 = float(np.sqrt(np.sum(e * e)) / (np.sqrt(np.sum(true_grid * true_grid)) + 1e-12))
        mx = float(np.max(np.abs(e)))
        return mse, rmse, mae, l2, mx

    mu = metrics(U_true, U_pred)
    mv = metrics(V_true, V_pred)
    mp = metrics(P_true, P_pred)
    mphi = metrics(PHI_true, PHI_pred)
    mt = metrics(TT_true, TT_pred)

    cert_uF, cert_vF, cert_phiF, cert_TF, cert_final = certificate_scan_and_adapt(step=MAX_STEPS + 1)

    metrics_path = os.path.join(logs_dir, "metrics_case10.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("PICS Case10 v3.1 metrics\n")
        f.write(f"train_time_s={train_time_s:.6f}\n")
        f.write(f"eval_time_s={eval_time_s:.6f}\n")
        f.write(f"total_time_s={total_time_s:.6f}\n")
        f.write(f"max_steps={MAX_STEPS}\n")
        f.write(f"Nf_final={Nf_current}\n")
        f.write(f"Nb={Nb}\n")
        f.write(f"final_cert_max={cert_final:.12e}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"dtype={torch.get_default_dtype()}\n")
        f.write(f"device={device}\n\n")
        f.write("field MSE RMSE MAE relL2 MaxError\n")
        f.write(f"u {mu[0]:.12e} {mu[1]:.12e} {mu[2]:.12e} {mu[3]:.12e} {mu[4]:.12e}\n")
        f.write(f"v {mv[0]:.12e} {mv[1]:.12e} {mv[2]:.12e} {mv[3]:.12e} {mv[4]:.12e}\n")
        f.write(f"p {mp[0]:.12e} {mp[1]:.12e} {mp[2]:.12e} {mp[3]:.12e} {mp[4]:.12e}\n")
        f.write(f"phi {mphi[0]:.12e} {mphi[1]:.12e} {mphi[2]:.12e} {mphi[3]:.12e} {mphi[4]:.12e}\n")
        f.write(f"T {mt[0]:.12e} {mt[1]:.12e} {mt[2]:.12e} {mt[3]:.12e} {mt[4]:.12e}\n")

    runtime_path = os.path.join(logs_dir, "runtime_case10.txt")
    with open(runtime_path, "w", encoding="utf-8") as f:
        f.write(f"train_time_s {train_time_s:.6f}\n")
        f.write(f"eval_time_s {eval_time_s:.6f}\n")
        f.write(f"total_time_s {total_time_s:.6f}\n")
        f.write(f"Nf_final {Nf_current}\n")
        f.write(f"Nb {Nb}\n")
        f.write(f"final_cert_max {cert_final:.12e}\n")

    hist = np.loadtxt(loss_hist_path, skiprows=1)
    steps = hist[:, 0]
    Ltot = hist[:, 2]
    Lphys = hist[:, 3]
    Lbc = hist[:, 4]
    Lcert = hist[:, 5]
    Lint = hist[:, 6]
    certs = hist[:, 7]

    plot_curve(os.path.join(figs_dir, "loss_linear.png"),
               steps, [Ltot, Lphys, Lbc, Lcert, Lint],
               ["L_total", "L_phys", "L_bc", "L_cert", "L_int"],
               ylog=False, xlabel="step", ylabel="loss", title="Loss history (linear)")
    plot_curve(os.path.join(figs_dir, "loss_log.png"),
               steps, [Ltot, Lphys, Lbc, Lcert, Lint],
               ["L_total", "L_phys", "L_bc", "L_cert", "L_int"],
               ylog=True, xlabel="step", ylabel="loss", title="Loss history (log)")
    plot_curve(os.path.join(figs_dir, "cert_linear.png"),
               steps, [certs], ["cert_max"], ylog=False,
               xlabel="step", ylabel="cert", title="Certificate history (linear)")
    plot_curve(os.path.join(figs_dir, "cert_log.png"),
               steps, [certs], ["cert_max"], ylog=True,
               xlabel="step", ylabel="cert", title="Certificate history (log)")

    readme_path = os.path.join(logs_dir, "README_case10.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("PICS Case10 v3.1 README\n\n")
        f.write("Constraints:\n")
        f.write("- No truth embedding in model forward; no truth function calls inside forward.\n")
        f.write("- Interior truth supervision points = 0.\n")
        f.write("- Training runs exactly 200000 steps with Adam only (no early stop).\n\n")
        f.write("Case10 PDE corrections:\n")
        f.write("- Pressure-gradient regularization: +c_p*Δ(∂x p), +c_p*Δ(∂y p) in momentum.\n")
        f.write("- Screened Poisson (Debye-Hückel): r_phi = -εΔφ + kappa_D^2 φ - rho_e.\n")
        f.write("  We define rho_e := -εΔφ* + kappa_D^2 φ* so manufactured φ* satisfies screened Poisson exactly.\n\n")
        f.write("PICS v3.1 mechanisms (same as Case01–09):\n")
        f.write("- Two-expert gated MoE with fixed gate g(x,y) and features (x,y,xi).\n")
        f.write("- Streamfunction psi provides u,v and enforces incompressibility.\n")
        f.write("- Constant affine scaling: p=1+P0 p_tilde, phi=PHI0 phi_tilde, T=T0+T1 T_tilde.\n")
        f.write("- Residual normalization based on RHS RMS.\n")
        f.write("- L_cert: log-mean-exp proxy for max tail of normalized residuals.\n")
        f.write("- L_int: interface consistency weighted by sech^2(xi).\n")
        f.write("- Hard buffer replay (70% batch) + certificate scan and adaptive point adding.\n\n")
        f.write("Sampling:\n")
        f.write("- Band sampling targets two stripe families; always returns (n,2).\n\n")
        f.write("Outputs:\n")
        f.write("- case10/figs: 15 jet plots + loss/cert curves.\n")
        f.write("- case10/data: 15 txt triplets x y value.\n")
        f.write("- case10/logs: metrics_case10.txt runtime_case10.txt loss_history.txt README_case10.txt.\n")

    print("\n=== FINAL SUMMARY ===")
    print(f"final_cert_max={cert_final:.6e}")
    print(f"total_time_s={total_time_s:.3f} (train={train_time_s:.3f}, eval={eval_time_s:.3f})")
    print(f"u relL2={mu[3]:.6e} MaxError={mu[4]:.6e}")
    print(f"v relL2={mv[3]:.6e} MaxError={mv[4]:.6e}")
    print(f"p relL2={mp[3]:.6e} MaxError={mp[4]:.6e}")
    print(f"phi relL2={mphi[3]:.6e} MaxError={mphi[4]:.6e}")
    print(f"T relL2={mt[3]:.6e} MaxError={mt[4]:.6e}")

    def list_files(d):
        return sorted(os.listdir(d))

    print("\ncase10/figs:")
    for name in list_files(figs_dir):
        print("  " + name)
    print("case10/data:")
    for name in list_files(data_dir):
        print("  " + name)
    print("case10/logs:")
    for name in list_files(logs_dir):
        print("  " + name)


if __name__ == "__main__":
    main()
