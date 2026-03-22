from __future__ import annotations

import gc
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec, ticker
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR_NAME = 'replot_like_uploaded_sample_v10'

ALGO_ORDER = ['PICS', 'DGM', 'DRM', 'PINN']
ALGO_COLORS = {
    'PICS': 'tab:red',
    'DGM': 'tab:blue',
    'DRM': 'tab:orange',
    'PINN': 'tab:green',
}

FIELD_PRIORITY = ['p', 'u', 'v', 'phi', 't']
FIELD_DISPLAY = {
    'p': r'$p$',
    'u': r'$u$',
    'v': r'$v$',
    'phi': r'$\phi$',
    't': r'$T$',
}

CASE_AXIS_LABELS = {
    'case 1': (r'$x$', r'$y$'),
    'case 9': (r'$x$', r'$y$'),
    'case 10': (r'$x$', r'$y$'),
}

CASE_DISPLAY_MAP = {
    'case 1': 'case 1',
    'case 9': 'case 2',
    'case 10': 'case 3',
}
CASE_OUTPUT_INDEX = {
    'case 1': 1,
    'case 9': 2,
    'case 10': 3,
}

PREFER_REPLOT_PICS_STYLE = True
PNG_DPI = 220
PDF_DPI = 220
CMAP_ALL = 'jet'
SAVE_TRANSPARENT = False
ROUND_DECIMALS_CANDIDATES = (12, 10, 8, 6)
FLOAT_PATTERN = re.compile(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?')
CASE_PATTERN = re.compile(r'^case\s*\d+$', re.IGNORECASE)
WORD_SPLIT = re.compile(r'[\s,;|]+')


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r'(\d+)', text)]


def canonical_field_name(name: str) -> str:
    s = name.strip().lower()
    return 't' if s == 'temperature' else s


def pretty_field_label(name: str) -> str:
    return FIELD_DISPLAY.get(canonical_field_name(name), name)


def pretty_field_title_bold(name: str) -> str:
    canonical = canonical_field_name(name)
    mapping = {
        'p': r'$\mathbf{p}$',
        'u': r'$\mathbf{u}$',
        'v': r'$\mathbf{v}$',
        'phi': r'$\boldsymbol{\phi}$',
        't': r'$\mathbf{T}$',
    }
    return mapping.get(canonical, pretty_field_label(name))


def ordered_case_fields(case_payload: dict) -> List[str]:
    ordered = [f for f in FIELD_PRIORITY if f in case_payload['fields']]
    ordered.extend([f for f in case_payload['fields'] if f not in ordered])
    return ordered


def dynamic_suite_figsize(nrows: int) -> Tuple[float, float]:
    return (19.2, max(11.8, 2.45 * nrows + 2.85))


def dynamic_error_figsize(nrows: int) -> Tuple[float, float]:
    return (12.0, max(7.2, 2.35 * nrows + 1.35))


def dynamic_exact_figsize(nfields: int) -> Tuple[float, float]:
    return (15.8, max(9.0, 2.35 * nfields + 3.25))


def dynamic_detail_figsize() -> Tuple[float, float]:
    return (13.6, 7.3)


def read_numeric_txt(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            nums = FLOAT_PATTERN.findall(s)
            if not nums:
                continue
            rows.append([float(v) for v in nums])
    if not rows:
        raise ValueError(f'文件中没有读到数字: {path}')
    widths = [len(r) for r in rows]
    target_width = max(set(widths), key=widths.count)
    rows = [r[:target_width] for r in rows if len(r) >= target_width]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


def choose_data_dir(run_dir: Path) -> Optional[Path]:
    data_dir = run_dir / 'data'
    if not data_dir.exists():
        return None
    replot_dir = data_dir / 'replot_pics_style'
    if PREFER_REPLOT_PICS_STYLE and replot_dir.exists() and any(replot_dir.glob('*.txt')):
        return replot_dir
    return data_dir


def resolve_run_dir(root: Path, case_name: str, algo: str) -> Optional[Path]:
    algo_dir = root / case_name / algo
    if not algo_dir.exists():
        return None
    if (algo_dir / 'data').exists():
        return algo_dir
    candidate_dirs = sorted(
        [p.parent for p in algo_dir.rglob('data') if p.is_dir()],
        key=lambda p: (len(p.parts), str(p).lower())
    )
    return candidate_dirs[0] if candidate_dirs else None


def collect_field_map(data_dir: Path) -> Dict[str, Dict[str, Path]]:
    out: Dict[str, Dict[str, Path]] = {}
    for p in sorted(data_dir.glob('*.txt'), key=lambda q: natural_key(q.name)):
        stem = p.stem
        if stem.startswith('replot'):
            continue
        lower = stem.lower()
        if lower.endswith('_true'):
            out.setdefault(canonical_field_name(stem[:-5]), {})['true'] = p
        elif lower.endswith('_pred'):
            out.setdefault(canonical_field_name(stem[:-5]), {})['pred'] = p
        elif lower.endswith('_maxerror'):
            out.setdefault(canonical_field_name(stem[:-9]), {})['maxerror'] = p
    return out


def try_build_rect_grid(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    for decimals in ROUND_DECIMALS_CANDIDATES:
        xr = np.round(x, decimals=decimals)
        yr = np.round(y, decimals=decimals)
        ux = np.unique(xr)
        uy = np.unique(yr)
        if ux.size * uy.size != x.size:
            continue
        ix = {v: i for i, v in enumerate(ux)}
        iy = {v: i for i, v in enumerate(uy)}
        Z = np.full((uy.size, ux.size), np.nan, dtype=float)
        count = np.zeros_like(Z, dtype=int)
        for xx, yy, zz in zip(xr, yr, z):
            i = ix[xx]
            j = iy[yy]
            if np.isnan(Z[j, i]):
                Z[j, i] = zz
            else:
                Z[j, i] += zz
            count[j, i] += 1
        if not np.all(count > 0):
            continue
        Z /= count
        X, Y = np.meshgrid(ux, uy)
        return {'mode': 'grid', 'x': X, 'y': Y, 'z': Z}
    return {'mode': 'scatter', 'x': x, 'y': y, 'z': z}


def read_field_file(path: Path):
    arr = read_numeric_txt(path)
    if arr.shape[1] < 3:
        raise ValueError(f'场数据至少应含三列 (x, y, value): {path}')
    return try_build_rect_grid(arr[:, 0], arr[:, 1], arr[:, 2])


def compute_abs_error_field(pred_field: dict, true_field: dict) -> dict:
    if pred_field['mode'] == 'grid' and true_field['mode'] == 'grid':
        if pred_field['z'].shape == true_field['z'].shape and np.allclose(pred_field['x'], true_field['x']) and np.allclose(pred_field['y'], true_field['y']):
            return {'mode': 'grid', 'x': pred_field['x'], 'y': pred_field['y'], 'z': np.abs(pred_field['z'] - true_field['z'])}
    raise ValueError('预测场与真解网格不一致，且缺少现成的 maxerror txt。')


def field_value_limits(field_list: List[dict]) -> Tuple[float, float]:
    vals = np.concatenate([np.ravel(fd['z']) for fd in field_list])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if math.isclose(vmin, vmax, rel_tol=1e-14, abs_tol=1e-14):
        eps = 1.0 if abs(vmin) < 1e-12 else 0.05 * abs(vmin)
        vmin -= eps
        vmax += eps
    return vmin, vmax


def format_colorbar(cb, vmin: float, vmax: float):
    maxabs = max(abs(vmin), abs(vmax))
    if maxabs != 0 and (maxabs < 1e-2 or maxabs >= 1e3):
        formatter = ticker.FormatStrFormatter('%.1e')
    else:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
    cb.formatter = formatter
    cb.update_ticks()
    cb.ax.tick_params(labelsize=6.3, length=2)


def plot_field(ax, field_data: dict, *, cmap: str, vmin: float, vmax: float,
               title: Optional[str] = None, xlabel: Optional[str] = None,
               ylabel: Optional[str] = None):
    if field_data['mode'] == 'grid':
        artist = ax.pcolormesh(field_data['x'], field_data['y'], field_data['z'], shading='auto', cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        xmin = float(np.nanmin(field_data['x']))
        xmax = float(np.nanmax(field_data['x']))
        ymin = float(np.nanmin(field_data['y']))
        ymax = float(np.nanmax(field_data['y']))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        artist = ax.tricontourf(field_data['x'], field_data['y'], field_data['z'], levels=120, cmap=cmap, vmin=vmin, vmax=vmax)
        xmin = float(np.nanmin(field_data['x']))
        xmax = float(np.nanmax(field_data['x']))
        ymin = float(np.nanmin(field_data['y']))
        ymax = float(np.nanmax(field_data['y']))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    xspan = max(abs(xmax - xmin), 1e-12)
    yspan = max(abs(ymax - ymin), 1e-12)
    ax.set_box_aspect(yspan / xspan)
    ax.set_aspect('auto')
    if title:
        ax.set_title(title, fontsize=9.5, fontweight='bold', pad=2.2)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.3, labelpad=1.0)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.3, labelpad=1.0)
    ax.tick_params(labelsize=6.6, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    return artist


def score_loss_file(path: Path) -> Tuple[int, int, int, str]:
    name = path.name.lower()
    parent = path.parent.name.lower()
    score = 0
    if 'total' in name:
        score += 50
    if name.startswith('loss_history'):
        score += 20
    if name == 'loss_history.txt':
        score += 10
    if parent == 'logs':
        score += 5
    return (-score, len(path.parts), len(name), str(path).lower())


def find_loss_history_file(run_dir: Path) -> Optional[Path]:
    patterns = ['total_loss*.txt', '*total*loss*.txt', 'loss_history*.txt', '*loss*.txt']
    candidates: List[Path] = []
    for pattern in patterns:
        for p in run_dir.rglob(pattern):
            if p.is_file() and p.suffix.lower() == '.txt':
                candidates.append(p)
    if not candidates:
        return None
    return sorted({p.resolve() for p in candidates}, key=score_loss_file)[0]


def _extract_header_candidates(lines: List[str]) -> Tuple[Optional[List[str]], List[List[float]]]:
    numeric_rows: List[List[float]] = []
    header_tokens: Optional[List[str]] = None
    for line in lines:
        s = line.strip()
        if not s:
            continue
        nums = FLOAT_PATTERN.findall(s)
        if nums:
            numeric_rows.append([float(v) for v in nums])
            continue
        tokens = [tok for tok in WORD_SPLIT.split(s.replace('\t', ' ').replace(':', ' ')) if tok]
        if len(tokens) >= 2 and header_tokens is None:
            header_tokens = [tok.strip().lower() for tok in tokens]
    return header_tokens, numeric_rows


def choose_curve_columns(arr: np.ndarray, header_tokens: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray]:
    ncol = arr.shape[1]
    x_idx = 0
    y_idx = 1 if ncol >= 2 else 0
    if header_tokens is not None and len(header_tokens) >= ncol:
        for i, token in enumerate(header_tokens[:ncol]):
            token = token.replace('-', '_')
            if any(key in token for key in ['iter', 'epoch', 'step']):
                x_idx = i
                break
        priority = ['total_loss', 'loss_total', 'overall_loss', 'total', 'mse_total', 'mse', 'loss']
        tokens = [tok.replace('-', '_') for tok in header_tokens[:ncol]]
        found = None
        for key in priority:
            for i, token in enumerate(tokens):
                if key == token or key in token:
                    found = i
                    break
            if found is not None:
                break
        if found is not None and found != x_idx:
            y_idx = found
    if ncol == 1:
        return np.arange(1, arr.shape[0] + 1, dtype=float), arr[:, 0]
    return arr[:, x_idx], arr[:, y_idx if y_idx != x_idx else 1]


def read_curve_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    lines = path.read_text(encoding='utf-8', errors='ignore').splitlines()
    header_tokens, numeric_rows = _extract_header_candidates(lines)
    if not numeric_rows:
        raise ValueError(f'没有在损失文件中读到数值: {path}')
    widths = [len(r) for r in numeric_rows]
    target_width = max(set(widths), key=widths.count)
    rows = [r[:target_width] for r in numeric_rows if len(r) >= target_width]
    arr = np.asarray(rows, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    x, y = choose_curve_columns(arr, header_tokens)
    good = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[good], dtype=float)
    y = np.asarray(y[good], dtype=float)
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    order = np.argsort(x)
    return x[order], np.clip(y[order], 1e-30, None)


def collect_case_payload(root: Path, case_name: str) -> Optional[dict]:
    algo_meta: Dict[str, dict] = {}
    for algo in ALGO_ORDER:
        run_dir = resolve_run_dir(root, case_name, algo)
        if run_dir is None:
            continue
        data_dir = choose_data_dir(run_dir)
        if data_dir is None:
            continue
        field_map = collect_field_map(data_dir)
        if not field_map:
            continue
        algo_meta[algo] = {'run_dir': run_dir, 'data_dir': data_dir, 'field_map': field_map, 'loss_file': find_loss_history_file(run_dir)}
    if not algo_meta:
        return None

    common_fields = None
    for algo in ALGO_ORDER:
        if algo not in algo_meta:
            continue
        fields = {k for k, v in algo_meta[algo]['field_map'].items() if 'true' in v and 'pred' in v}
        common_fields = fields if common_fields is None else (common_fields & fields)
    if not common_fields:
        return None

    ordered_fields = [f for f in FIELD_PRIORITY if f in common_fields]
    ordered_fields.extend(sorted([f for f in common_fields if f not in ordered_fields], key=natural_key))

    fields_payload: Dict[str, dict] = {}
    for field in ordered_fields:
        true_field = None
        preds: Dict[str, dict] = {}
        errs: Dict[str, dict] = {}
        files_map: Dict[str, Dict[str, Path]] = {}
        for algo in ALGO_ORDER:
            if algo not in algo_meta:
                continue
            fmap = algo_meta[algo]['field_map'].get(field)
            if not fmap or 'true' not in fmap or 'pred' not in fmap:
                continue
            true_now = read_field_file(fmap['true'])
            pred_now = read_field_file(fmap['pred'])
            err_now = read_field_file(fmap['maxerror']) if 'maxerror' in fmap else compute_abs_error_field(pred_now, true_now)
            if true_field is None:
                true_field = true_now
            preds[algo] = pred_now
            errs[algo] = err_now
            files_map[algo] = fmap
        if true_field is None or len(preds) == 0:
            continue
        vmin, vmax = field_value_limits([true_field] + [preds[a] for a in ALGO_ORDER if a in preds])
        emin, emax = field_value_limits([errs[a] for a in ALGO_ORDER if a in errs])
        emin = 0.0
        if math.isclose(emin, emax, rel_tol=1e-14, abs_tol=1e-14):
            emax += 1.0 if emax == 0 else 0.05 * abs(emax)
        fields_payload[field] = {
            'label': pretty_field_label(field),
            'true': true_field,
            'preds': preds,
            'errs': errs,
            'files': files_map,
            'vmin': vmin,
            'vmax': vmax,
            'emin': emin,
            'emax': emax,
        }
    if not fields_payload:
        return None

    losses = {}
    loss_files = {}
    for algo in ALGO_ORDER:
        if algo not in algo_meta:
            continue
        lf = algo_meta[algo]['loss_file']
        if lf is None:
            continue
        try:
            x, y = read_curve_file(lf)
            if x.size > 0:
                losses[algo] = (x, y)
                loss_files[algo] = lf
        except Exception:
            pass

    display_case_name = CASE_DISPLAY_MAP.get(case_name, case_name)
    output_index = CASE_OUTPUT_INDEX.get(case_name)
    if output_index is None:
        m = re.search(r'\d+', case_name)
        output_index = int(m.group()) if m else 0

    return {
        'case_name': case_name,
        'display_case_name': display_case_name,
        'output_index': output_index,
        'axis_labels': CASE_AXIS_LABELS.get(case_name, (r'$x$', r'$y$')),
        'fields': fields_payload,
        'losses': losses,
        'loss_files': loss_files,
    }


def prepare_loss_curves(case_payload: dict) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    curves = []
    for algo in ALGO_ORDER:
        if algo in case_payload['losses']:
            curves.append((algo, case_payload['losses'][algo][0], case_payload['losses'][algo][1]))
    return curves


def compute_zoom_window(curves: List[Tuple[str, np.ndarray, np.ndarray]]) -> Optional[Tuple[float, float, float, float]]:
    if not curves:
        return None
    xmin = min(float(np.min(x)) for _, x, _ in curves)
    xmax = max(float(np.max(x)) for _, x, _ in curves)
    if math.isclose(xmin, xmax):
        return None
    x0 = xmin + 0.85 * (xmax - xmin)
    ys = []
    for _, x, y in curves:
        mask = x >= x0
        if np.any(mask):
            ys.append(y[mask])
    if not ys:
        return None
    yy = np.concatenate(ys)
    yy = yy[np.isfinite(yy) & (yy > 0)]
    if yy.size == 0:
        return None
    y0 = float(np.min(yy))
    y1 = float(np.max(yy))
    if math.isclose(y0, y1, rel_tol=1e-14, abs_tol=1e-14):
        y0 *= 0.85
        y1 *= 1.15
    else:
        y0 *= 0.85
        y1 *= 1.15
    return x0, xmax, max(y0, 1e-30), max(y1, 1e-29)


def draw_loss_panel(ax, case_payload: dict):
    curves = prepare_loss_curves(case_payload)
    any_curve = False
    for algo, x, y in curves:
        ax.plot(x, y, lw=1.15, color=ALGO_COLORS[algo], label=algo)
        any_curve = True
    ax.set_yscale('log')
    ax.set_xlabel('')
    ax.set_ylabel('Loss', fontsize=8.3)
    ax.tick_params(labelsize=7.2, length=2)
    ax.grid(True, which='both', linestyle='-', linewidth=0.42, alpha=0.33)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
    if not any_curve:
        ax.text(0.5, 0.5, 'No loss history', ha='center', va='center', fontsize=8.2, transform=ax.transAxes)
        return
    ax.legend(fontsize=6.7, frameon=True, loc='upper right', ncol=4)
    zoom = compute_zoom_window(curves)
    if zoom is None:
        return
    x0, x1, y0, y1 = zoom
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='black', linewidth=0.65, linestyle='--'))
    axins = ax.inset_axes([0.655, 0.28, 0.31, 0.56])
    for algo, x, y in curves:
        axins.plot(x, y, lw=0.98, color=ALGO_COLORS[algo])
    axins.set_yscale('log')
    axins.set_xlim(x0, x1)
    axins.set_ylim(y0, y1)
    axins.grid(True, which='both', linestyle='-', linewidth=0.32, alpha=0.28)
    axins.tick_params(labelsize=5.7, length=1.6)
    for spine in axins.spines.values():
        spine.set_linewidth(0.52)
    ax.indicate_inset_zoom(axins, edgecolor='black', alpha=0.85)


def png_to_pdf(png_path: Path, pdf_path: Path):
    img = Image.open(png_path).convert('RGB')
    img.save(pdf_path, 'PDF', resolution=PDF_DPI)
    img.close()


def save_figure(fig, base: Path, *, save_png: bool = True, save_pdf: bool = True):
    png_path = base.with_suffix('.png')
    if save_png or save_pdf:
        fig.savefig(png_path, dpi=PNG_DPI, transparent=SAVE_TRANSPARENT)
    if save_pdf:
        png_to_pdf(png_path, base.with_suffix('.pdf'))
    fig.canvas.draw_idle()


def field_to_xyz(field_data: dict) -> np.ndarray:
    if field_data['mode'] == 'grid':
        x = np.ravel(field_data['x'])
        y = np.ravel(field_data['y'])
        z = np.ravel(field_data['z'])
    else:
        x = np.ravel(field_data['x'])
        y = np.ravel(field_data['y'])
        z = np.ravel(field_data['z'])
    return np.column_stack([x, y, z])


def write_curve_block(fp, name: str, x: np.ndarray, y: np.ndarray):
    fp.write(f'# {name}\n')
    fp.write('iteration loss\n')
    data = np.column_stack([x, y])
    np.savetxt(fp, data, fmt='%.12e')
    fp.write('\n')


def write_field_block(fp, name: str, field_data: dict):
    fp.write(f'# {name}\n')
    fp.write('x y value\n')
    np.savetxt(fp, field_to_xyz(field_data), fmt='%.12e')
    fp.write('\n')


def export_combined_figure_data(case_payload: dict, base: Path):
    with base.with_suffix('.txt').open('w', encoding='utf-8') as fp:
        fp.write(f'# figure data | {case_payload["display_case_name"]} | exact+pred+maxerror+loss\n\n')
        for algo in ALGO_ORDER:
            if algo in case_payload['losses']:
                x, y = case_payload['losses'][algo]
                write_curve_block(fp, f'loss | algo={algo}', x, y)
        for field in ordered_case_fields(case_payload):
            payload = case_payload['fields'][field]
            write_field_block(fp, f'field=true | var={field}', payload['true'])
            for algo in ALGO_ORDER:
                write_field_block(fp, f'field=pred | var={field} | algo={algo}', payload['preds'][algo])
            for algo in ALGO_ORDER:
                write_field_block(fp, f'field=maxerror | var={field} | algo={algo}', payload['errs'][algo])


def export_maxerror_only_data(case_payload: dict, base: Path):
    with base.with_suffix('.txt').open('w', encoding='utf-8') as fp:
        fp.write(f'# figure data | {case_payload["display_case_name"]} | maxerror only\n\n')
        for field in ordered_case_fields(case_payload):
            payload = case_payload['fields'][field]
            for algo in ALGO_ORDER:
                write_field_block(fp, f'field=maxerror | var={field} | algo={algo}', payload['errs'][algo])


def export_exact_with_loss_data(case_payload: dict, base: Path):
    with base.with_suffix('.txt').open('w', encoding='utf-8') as fp:
        fp.write(f'# figure data | {case_payload["display_case_name"]} | exact+pred+loss\n\n')
        for algo in ALGO_ORDER:
            if algo in case_payload['losses']:
                x, y = case_payload['losses'][algo]
                write_curve_block(fp, f'loss | algo={algo}', x, y)
        for field in ordered_case_fields(case_payload):
            payload = case_payload['fields'][field]
            write_field_block(fp, f'field=true | var={field}', payload['true'])
            for algo in ALGO_ORDER:
                write_field_block(fp, f'field=pred | var={field} | algo={algo}', payload['preds'][algo])


def export_detail_data(case_payload: dict, field: str, base: Path):
    payload = case_payload['fields'][field]
    with base.with_suffix('.txt').open('w', encoding='utf-8') as fp:
        fp.write(f'# figure data | {case_payload["display_case_name"]} | detail | var={field}\n\n')
        write_field_block(fp, f'field=true | var={field}', payload['true'])
        for algo in ALGO_ORDER:
            write_field_block(fp, f'field=pred | var={field} | algo={algo}', payload['preds'][algo])
        for algo in ALGO_ORDER:
            write_field_block(fp, f'field=maxerror | var={field} | algo={algo}', payload['errs'][algo])


def draw_bold_field_label(ax, label: str, *, fontsize: float = 12.8, x: float = -0.34):
    ax.text(
        x, 0.50, label,
        transform=ax.transAxes,
        rotation=90,
        va='center', ha='center',
        fontsize=fontsize,
        fontweight='bold',
    )


def draw_bold_row_label(ax, label: str):
    ax.text(
        -0.42, 0.50, label,
        transform=ax.transAxes,
        rotation=90,
        va='center', ha='center',
        fontsize=10.8,
        fontweight='bold',
    )


def draw_case_header(fig, case_payload: dict, subtitle: Optional[str] = None):
    text = case_payload['display_case_name'] if subtitle is None else f"{case_payload['display_case_name']} | {subtitle}"
    fig.text(0.015, 0.985, text, fontsize=12.2, fontweight='bold', va='top')


def draw_suite_plus_error_with_loss_figure(case_payload: dict):
    fields = ordered_case_fields(case_payload)
    nrows = len(fields)
    xlab, ylab = case_payload['axis_labels']
    fig = plt.figure(figsize=dynamic_suite_figsize(nrows))
    outer = gridspec.GridSpec(
        nrows + 1, 12, figure=fig,
        height_ratios=[0.95] + [1.0] * nrows,
        width_ratios=[1, 1, 1, 1, 1, 0.078, 0.16, 1, 1, 1, 1, 0.078],
        wspace=0.20, hspace=0.28
    )

    loss_ax = fig.add_subplot(outer[0, 0:11])
    draw_loss_panel(loss_ax, case_payload)
    fig.add_subplot(outer[0, 11]).axis('off')

    left_titles = ['Exact'] + ALGO_ORDER
    right_titles = ALGO_ORDER

    for row, field in enumerate(fields, start=1):
        payload = case_payload['fields'][field]
        left_axes = [fig.add_subplot(outer[row, c]) for c in range(5)]
        cax_field = fig.add_subplot(outer[row, 5])
        fig.add_subplot(outer[row, 6]).axis('off')
        right_axes = [fig.add_subplot(outer[row, c]) for c in range(7, 11)]
        cax_err = fig.add_subplot(outer[row, 11])

        artist_field = plot_field(left_axes[0], payload['true'], cmap=CMAP_ALL, vmin=payload['vmin'], vmax=payload['vmax'], title=left_titles[0] if row == 1 else None, xlabel=xlab if row == nrows else None, ylabel=ylab)
        draw_bold_field_label(left_axes[0], payload['label'])
        for j, algo in enumerate(ALGO_ORDER, start=1):
            artist_field = plot_field(left_axes[j], payload['preds'][algo], cmap=CMAP_ALL, vmin=payload['vmin'], vmax=payload['vmax'], title=left_titles[j] if row == 1 else None, xlabel=xlab if row == nrows else None, ylabel=None)
            left_axes[j].set_yticklabels([])

        artist_err = None
        for j, algo in enumerate(ALGO_ORDER):
            artist_err = plot_field(right_axes[j], payload['errs'][algo], cmap=CMAP_ALL, vmin=payload['emin'], vmax=payload['emax'], title=right_titles[j] if row == 1 else None, xlabel=xlab if row == nrows else None, ylabel=ylab if j == 0 else None)
            if j > 0:
                right_axes[j].set_yticklabels([])

        if row != nrows:
            for ax in left_axes + right_axes:
                ax.set_xticklabels([])

        cb1 = fig.colorbar(artist_field, cax=cax_field)
        cb2 = fig.colorbar(artist_err, cax=cax_err)
        format_colorbar(cb1, payload['vmin'], payload['vmax'])
        format_colorbar(cb2, payload['emin'], payload['emax'])

    draw_case_header(fig, case_payload)
    fig.subplots_adjust(left=0.052, right=0.974, top=0.948, bottom=0.06)
    return fig


def draw_exact_with_loss_figure(case_payload: dict):
    fields = ordered_case_fields(case_payload)
    nfields = len(fields)
    xlab, ylab = case_payload['axis_labels']
    fig = plt.figure(figsize=dynamic_exact_figsize(nfields))
    outer = gridspec.GridSpec(
        nfields + 1, 7, figure=fig,
        height_ratios=[0.92] + [1.0] * nfields,
        width_ratios=[1, 1, 1, 1, 1, 0.08, 0.11],
        wspace=0.18, hspace=0.26
    )

    loss_ax = fig.add_subplot(outer[0, 0:6])
    draw_loss_panel(loss_ax, case_payload)
    fig.add_subplot(outer[0, 6]).axis('off')

    col_titles = ['Exact'] + ALGO_ORDER

    for row, field in enumerate(fields, start=1):
        payload = case_payload['fields'][field]
        axes = [fig.add_subplot(outer[row, c]) for c in range(5)]
        cax = fig.add_subplot(outer[row, 5])
        fig.add_subplot(outer[row, 6]).axis('off')

        field_stack = [payload['true']] + [payload['preds'][algo] for algo in ALGO_ORDER]
        artist = None
        for j, field_data in enumerate(field_stack):
            title = col_titles[j] if row == 1 else None
            artist = plot_field(
                axes[j], field_data, cmap=CMAP_ALL,
                vmin=payload['vmin'], vmax=payload['vmax'],
                title=title,
                xlabel=xlab if row == nfields else None,
                ylabel=ylab if j == 0 else None,
            )
            if j > 0:
                axes[j].set_yticklabels([])
        if row != nfields:
            for ax in axes:
                ax.set_xticklabels([])

        draw_bold_field_label(axes[0], pretty_field_title_bold(field), fontsize=13.2, x=-0.37)
        cb = fig.colorbar(artist, cax=cax)
        format_colorbar(cb, payload['vmin'], payload['vmax'])
        cax.tick_params(labelsize=5.9, length=1.8, pad=1.7)

    draw_case_header(fig, case_payload, subtitle='Loss + field suite')
    fig.subplots_adjust(left=0.094, right=0.972, top=0.952, bottom=0.072)
    return fig


def draw_maxerror_only_figure(case_payload: dict):
    fields = ordered_case_fields(case_payload)
    nrows = len(fields)
    xlab, ylab = case_payload['axis_labels']
    fig = plt.figure(figsize=dynamic_error_figsize(nrows))
    outer = gridspec.GridSpec(
        nrows, 6, figure=fig,
        width_ratios=[1, 1, 1, 1, 0.08, 0.11],
        wspace=0.20, hspace=0.28
    )
    for row, field in enumerate(fields):
        payload = case_payload['fields'][field]
        axes = [fig.add_subplot(outer[row, c]) for c in range(4)]
        cax = fig.add_subplot(outer[row, 4])
        fig.add_subplot(outer[row, 5]).axis('off')
        artist = None
        for j, algo in enumerate(ALGO_ORDER):
            artist = plot_field(
                axes[j], payload['errs'][algo], cmap=CMAP_ALL,
                vmin=payload['emin'], vmax=payload['emax'],
                title=algo if row == 0 else None,
                xlabel=xlab if row == nrows - 1 else None,
                ylabel=ylab if j == 0 else None
            )
            if j > 0:
                axes[j].set_yticklabels([])
        draw_bold_field_label(axes[0], pretty_field_title_bold(field), fontsize=13.2, x=-0.37)
        if row != nrows - 1:
            for ax in axes:
                ax.set_xticklabels([])
        cb = fig.colorbar(artist, cax=cax)
        format_colorbar(cb, payload['emin'], payload['emax'])
        cax.tick_params(labelsize=6.0, length=1.8, pad=1.7)
    draw_case_header(fig, case_payload, subtitle='Max error only')
    fig.subplots_adjust(left=0.095, right=0.972, top=0.932, bottom=0.075)
    return fig


def draw_detail_field_figure(case_payload: dict, field: str, output_dir: Path):
    payload = case_payload['fields'][field]
    xlab, ylab = case_payload['axis_labels']
    fig = plt.figure(figsize=dynamic_detail_figsize())
    outer = gridspec.GridSpec(2, 7, figure=fig, width_ratios=[1, 1, 1, 1, 1, 0.08, 0.08], wspace=0.24, hspace=0.24)
    top_axes = [fig.add_subplot(outer[0, i]) for i in range(5)]
    err_axes = [fig.add_subplot(outer[1, i]) for i in range(1, 5)]
    cax_field = fig.add_subplot(outer[0, 5])
    cax_err = fig.add_subplot(outer[1, 5])
    fig.add_subplot(outer[1, 0]).axis('off')
    fig.add_subplot(outer[:, 6]).axis('off')

    titles = ['Exact'] + ALGO_ORDER
    artist_field = plot_field(top_axes[0], payload['true'], cmap=CMAP_ALL, vmin=payload['vmin'], vmax=payload['vmax'], title=titles[0], ylabel=ylab)
    for j, algo in enumerate(ALGO_ORDER, start=1):
        artist_field = plot_field(top_axes[j], payload['preds'][algo], cmap=CMAP_ALL, vmin=payload['vmin'], vmax=payload['vmax'], title=titles[j])
        top_axes[j].set_yticklabels([])
    for ax in top_axes:
        ax.set_xticklabels([])

    artist_err = None
    for j, algo in enumerate(ALGO_ORDER):
        artist_err = plot_field(err_axes[j], payload['errs'][algo], cmap=CMAP_ALL, vmin=payload['emin'], vmax=payload['emax'], xlabel=xlab, ylabel=ylab if j == 0 else None)
        if j > 0:
            err_axes[j].set_yticklabels([])

    cb1 = fig.colorbar(artist_field, cax=cax_field)
    cb2 = fig.colorbar(artist_err, cax=cax_err)
    format_colorbar(cb1, payload['vmin'], payload['vmax'])
    format_colorbar(cb2, payload['emin'], payload['emax'])
    fig.text(0.02, 0.985, f"{case_payload['display_case_name']} | field: {payload['label']}", fontsize=10.6, fontweight='bold', va='top')
    fig.text(0.075, 0.275, 'Max error', fontsize=8.3, fontweight='bold', rotation=90, va='center')
    fig.subplots_adjust(left=0.06, right=0.972, top=0.91, bottom=0.09)

    base = output_dir / f"case{case_payload['output_index']:02d}_{field}_detail"
    save_figure(fig, base, save_png=True, save_pdf=False)
    plt.close(fig)
    gc.collect()
    export_detail_data(case_payload, field, base)


def draw_suite_plus_error_with_loss_case_figure(case_payload: dict, output_dir: Path):
    base = output_dir / f"case{case_payload['output_index']:02d}_suite_plus_error_with_loss"
    fig = draw_suite_plus_error_with_loss_figure(case_payload)
    save_figure(fig, base)
    plt.close(fig)
    gc.collect()
    export_combined_figure_data(case_payload, base)


def draw_maxerror_only_case_figure(case_payload: dict, output_dir: Path):
    base = output_dir / f"case{case_payload['output_index']:02d}_maxerror_only"
    fig = draw_maxerror_only_figure(case_payload)
    save_figure(fig, base)
    plt.close(fig)
    gc.collect()
    export_maxerror_only_data(case_payload, base)


def draw_exact_with_loss_case_figure(case_payload: dict, output_dir: Path):
    base = output_dir / f"case{case_payload['output_index']:02d}_field_suite_with_loss"
    fig = draw_exact_with_loss_figure(case_payload)
    save_figure(fig, base)
    plt.close(fig)
    gc.collect()
    export_exact_with_loss_data(case_payload, base)


def build_multipage_pdf_from_pngs(png_paths: List[Path], pdf_path: Path):
    images = [Image.open(p).convert('RGB') for p in png_paths if p.exists()]
    if not images:
        return
    first, rest = images[0], images[1:]
    first.save(pdf_path, 'PDF', resolution=PDF_DPI, save_all=True, append_images=rest)
    for img in images:
        img.close()


def save_suite_error_with_loss_multipage_pdf(case_payloads: List[dict], output_dir: Path):
    pngs = [output_dir / f"case{cp['output_index']:02d}_suite_plus_error_with_loss.png" for cp in case_payloads]
    build_multipage_pdf_from_pngs(pngs, output_dir / 'all_cases_suite_plus_error_with_loss.pdf')


def save_maxerror_multipage_pdf(case_payloads: List[dict], output_dir: Path):
    pngs = [output_dir / f"case{cp['output_index']:02d}_maxerror_only.png" for cp in case_payloads]
    build_multipage_pdf_from_pngs(pngs, output_dir / 'all_cases_maxerror_only.pdf')


def save_exact_with_loss_multipage_pdf(case_payloads: List[dict], output_dir: Path):
    pngs = [output_dir / f"case{cp['output_index']:02d}_field_suite_with_loss.png" for cp in case_payloads]
    build_multipage_pdf_from_pngs(pngs, output_dir / 'all_cases_field_suite_with_loss.pdf')


def write_manifest(case_payloads: List[dict], output_dir: Path):
    lines = ['replot manifest\n', f'root = {ROOT_DIR}\n', f'output = {output_dir}\n\n']
    for case_payload in case_payloads:
        lines.append(f"[{case_payload['display_case_name']}]\n")
        lines.append(f"source_case_dir = {case_payload['case_name']}\n")
        for algo in ALGO_ORDER:
            if algo in case_payload['loss_files']:
                lines.append(f'loss[{algo}] = {case_payload["loss_files"][algo]}\n')
        for field in ordered_case_fields(case_payload):
            lines.append(f'field = {field}\n')
            for algo in ALGO_ORDER:
                fmap = case_payload['fields'][field]['files'].get(algo)
                if fmap:
                    for key in ['true', 'pred', 'maxerror']:
                        if key in fmap:
                            lines.append(f'  {algo}.{key} = {fmap[key]}\n')
        lines.append('\n')
    (output_dir / 'manifest.txt').write_text(''.join(lines), encoding='utf-8')


def discover_cases(root: Path) -> List[str]:
    return sorted([p.name for p in root.iterdir() if p.is_dir() and CASE_PATTERN.match(p.name)], key=natural_key)


def main():
    plt.rcParams.update({
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'mathtext.fontset': 'dejavusans',
    })
    output_dir = ROOT_DIR / OUTPUT_DIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    case_names = discover_cases(ROOT_DIR)
    if not case_names:
        raise FileNotFoundError(f'在 {ROOT_DIR} 下没有找到 case 目录。')

    case_payloads: List[dict] = []
    for case_name in case_names:
        payload = collect_case_payload(ROOT_DIR, case_name)
        if payload is None:
            print(f'[skip] {case_name}')
            continue
        case_payloads.append(payload)
        draw_suite_plus_error_with_loss_case_figure(payload, output_dir)
        draw_maxerror_only_case_figure(payload, output_dir)
        draw_exact_with_loss_case_figure(payload, output_dir)
        for field in ordered_case_fields(payload):
            draw_detail_field_figure(payload, field, output_dir)
        gc.collect()
        print(f'[done] {payload["display_case_name"]} <= {case_name}')

    if not case_payloads:
        raise RuntimeError('没有生成任何图。')

    case_payloads.sort(key=lambda cp: cp['output_index'])
    save_suite_error_with_loss_multipage_pdf(case_payloads, output_dir)
    save_maxerror_multipage_pdf(case_payloads, output_dir)
    save_exact_with_loss_multipage_pdf(case_payloads, output_dir)
    write_manifest(case_payloads, output_dir)
    print(f'全部完成，输出目录: {output_dir}')


if __name__ == '__main__':
    main()
