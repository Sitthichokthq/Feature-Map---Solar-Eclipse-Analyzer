"""
analysis.py — Computation backend
Feature Map & Solar Eclipse Analyzer + CMD Plot
ครูที่ปรึกษา: วิทชภณ พวงแก้ว
Developed by: Sitthichokthq
"""

import cv2
import numpy as np
import math
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d, maximum_filter, label as ndlabel

try:
    from skimage.measure import ransac, CircleModel
    HAS_SKIMAGE = True
except ModuleNotFoundError:
    HAS_SKIMAGE = False

# ── Astronomical Constants ──────────────────────────────────────────────────
R_e              = 6371.0
R_s              = 696340.0
D_s              = 149597870.7
D_m              = 384400.0
MOON_DIAMETER_KM = 3474.8


# ── Image Filters ────────────────────────────────────────────────────────────

FILTERS = ["Cross-Correlation", "Sobel Edge", "Laplacian", "Sharpen"]


def apply_filter(gray: np.ndarray, mode: str) -> np.ndarray:
    """Apply the selected convolution filter to a grayscale image."""
    kernels = {
        "Cross-Correlation": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        "Sharpen":           np.array([[0, -1, 0],  [-1, 5, -1], [0, -1, 0]]),
    }
    if mode in kernels:
        return cv2.filter2D(gray, -1, kernels[mode])
    if mode == "Sobel Edge":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sx, sy).astype(np.uint8)
    if mode == "Laplacian":
        return cv2.Laplacian(gray, cv2.CV_8U)
    return gray


# ── Circle Fitting ───────────────────────────────────────────────────────────

def fit_circle_taubin(xs, ys):
    """Taubin algebraic circle fit — robust, no initial guess needed."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3:
        raise ValueError("need >= 3 pts")
    mx, my = xs.mean(), ys.mean()
    u, v   = xs - mx, ys - my
    Suu, Svv, Suv = (u**2).sum(), (v**2).sum(), (u * v).sum()
    A = np.array([[Suu, Suv], [Suv, Svv]])
    b = np.array([0.5 * (u**3 + u * v**2).sum(),
                  0.5 * (v**3 + v * u**2).sum()])
    try:
        uc, vc = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        uc, vc = 0.0, 0.0
    return uc + mx, vc + my, float(np.sqrt(uc**2 + vc**2 + (Suu + Svv) / len(xs)))


def fit_circle_constrained(p, expected_r=None, cx_moon=None, cy_moon=None, r_moon=None):
    """Least-squares circle fit with optional radius / centre constraints."""
    def calc_R(c):
        return np.sqrt((p[:, 0] - c[0])**2 + (p[:, 1] - c[1])**2)

    def f(c):
        residuals = calc_R(c) - c[2]
        r_pen = (c[2] - expected_r) * 0.1 if expected_r else 0.0
        c_pen = 0.0
        if cx_moon and cy_moon and r_moon:
            d = np.sqrt((c[0] - cx_moon)**2 + (c[1] - cy_moon)**2)
            if d < r_moon * 1.5:
                c_pen = (r_moon * 1.5 - d) * 10.0
        return np.append(residuals, [r_pen * len(p), c_pen * len(p) // 2])

    ce  = np.mean(p, axis=0)
    gr  = expected_r if expected_r else calc_R(np.append(ce, 0)).mean()
    bnd = ([-np.inf, -np.inf, expected_r * 0.85],
           [np.inf,  np.inf,  expected_r * 1.15]) if expected_r else (-np.inf, np.inf)
    res = least_squares(f, [ce[0], ce[1], gr], bounds=bnd)
    return res.x[0], res.x[1], res.x[2]


# ── Limb Darkening ───────────────────────────────────────────────────────────

def correct_limb_darkening(brightness, r_vals, r_moon, u=0.8):
    """Apply linear limb-darkening correction (Eddington law)."""
    r_frac    = np.clip(r_vals / r_moon, 0, 0.9999)
    cos_theta = np.sqrt(1 - r_frac**2)
    corr      = 1 - u * (1 - cos_theta)
    I = brightness.copy()
    vm = r_vals < 0.98 * r_moon
    I[vm] = brightness[vm] / corr[vm]
    return I


# ── Multi-frame Align & Stack ────────────────────────────────────────────────

def align_and_median_combine(paths):
    """
    Load multiple eclipse frames, align to first frame's moon centre/radius,
    then median-stack for SNR improvement.

    Returns: (stacked_bgr_image, (cx, cy, r_moon)) or (None, None) on failure.
    """
    images, crs = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 100,
                             param1=100, param2=50,
                             minRadius=50, maxRadius=int(min(img.shape[:2]) / 2))
        if c is not None:
            x = np.uint16(np.around(c))[0, 0]
            crs.append((x[0], x[1], x[2]))
            images.append(img)

    if not images:
        return None, None

    mr  = int(np.median([c[2] for c in crs]))
    tcx = images[0].shape[1] // 2
    tcy = images[0].shape[0] // 2
    aligned = []
    for img, (cx, cy, r) in zip(images, crs):
        s = mr / r
        if s != 1.0:
            img = cv2.resize(img, None, fx=s, fy=s,
                             interpolation=cv2.INTER_LANCZOS4)
            cx, cy = int(cx * s), int(cy * s)
        M = np.float32([[1, 0, tcx - cx], [0, 1, tcy - cy]])
        aligned.append(cv2.warpAffine(img, M,
                                      (images[0].shape[1], images[0].shape[0])))

    stacked = np.median(np.stack(aligned, 0), 0).astype(np.uint8)
    return stacked, (tcx, tcy, mr)


# ── Eclipse Ring Detection ───────────────────────────────────────────────────

def detect_rings(img_bgr, cx, cy, r_moon, limb_u=0.8):
    """
    Detect umbra outer edge and Moon inner limb from a total/annular eclipse image.

    Returns a dict with centres, radii, theoretical values, and quality metrics.
    """
    km_pp  = MOON_DIAMETER_KM / (2 * r_moon)
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray, (7, 7), 0)
    eq     = cv2.createCLAHE(2.0, (8, 8)).apply(blur)

    angles = np.arange(0, 360, 0.5)
    max_r  = int(2.5 * r_moon)
    r_vals = np.arange(0, max_r)

    outer_pts, inner_pts = [], []
    outer_profiles, inner_profiles = [], []
    best_outer_ray = best_inner_ray = None
    min_outer = min_inner = 0

    for angle in angles:
        rad = np.deg2rad(angle)
        xc  = cx + r_vals * np.cos(rad)
        yc  = cy + r_vals * np.sin(rad)
        ok  = (xc >= 0) & (xc < img_bgr.shape[1]) & \
              (yc >= 0) & (yc < img_bgr.shape[0])
        if not np.any(ok):
            continue
        xv, yv, rv = xc[ok], yc[ok], r_vals[ok]
        xi = np.clip(np.round(xv).astype(int), 0, img_bgr.shape[1] - 1)
        yi = np.clip(np.round(yv).astype(int), 0, img_bgr.shape[0] - 1)
        br = correct_limb_darkening(eq[yi, xi].astype(float), rv, r_moon, limb_u)

        wl = min(31, len(br) - (1 if len(br) % 2 == 0 else 0))
        sm = savgol_filter(br, wl, 3) if wl > 3 else br
        gd = np.gradient(sm)
        sg = cv2.GaussianBlur(gd.reshape(-1, 1), (15, 15), 0).flatten()

        # ── Inner ring (moon limb) ────────────────────────
        inner_zone = (rv >= r_moon * 0.6) & (rv <= r_moon * 1.1)
        iz = np.where(inner_zone)[0]
        if len(iz) >= 5:
            ig = sg[iz]
            ii = np.argmin(ig)
            if ig[ii] < -0.3:
                bi = iz[ii]
                sw, ew = max(0, bi - 5), min(len(gd), bi + 6)
                if ew - sw >= 3:
                    fi  = interp1d(np.arange(sw, ew), gd[sw:ew],
                                   kind='cubic', fill_value='extrapolate')
                    xn  = np.linspace(sw, ew - 1, (ew - sw) * 100)
                    rfr = rv[0] + xn[np.argmin(fi(xn))]
                    inner_pts.append([cx + rfr * np.cos(rad),
                                      cy + rfr * np.sin(rad)])
                    inner_profiles.append(sm.copy())
                    if ig[ii] < min_inner:
                        min_inner = ig[ii]
                        best_inner_ray = (sm.copy(), gd.copy(), rv.copy(), rfr)

        # ── Outer ring (umbra edge) ───────────────────────
        mask = np.ones_like(sg, dtype=bool)
        es, ee = int(0.95 * r_moon), int(1.05 * r_moon)
        if ee < len(mask):
            mask[es:ee] = False
        mask[:int(0.15 * r_moon)] = False
        vi = np.where(mask)[0]
        if len(vi) < 10:
            continue
        mg = sg[vi]
        mi = np.argmin(mg)
        if mg[mi] > -0.5:
            continue
        bri = vi[mi]
        sw, ew = max(0, bri - 5), min(len(gd), bri + 6)
        if ew - sw < 3:
            continue
        fo  = interp1d(np.arange(sw, ew), gd[sw:ew],
                       kind='cubic', fill_value='extrapolate')
        xn  = np.linspace(sw, ew - 1, (ew - sw) * 100)
        rfr = rv[0] + xn[np.argmin(fo(xn))]
        outer_pts.append([cx + rfr * np.cos(rad),
                          cy + rfr * np.sin(rad)])
        outer_profiles.append(sm.copy())
        if mg[mi] < min_outer:
            min_outer = mg[mi]
            best_outer_ray = (sm.copy(), gd.copy(), rv.copy(), rfr)

    outer_pts = np.array(outer_pts) if outer_pts else np.empty((0, 2))
    inner_pts = np.array(inner_pts) if inner_pts else np.empty((0, 2))

    # ── Theoretical umbra radius ──────────────────────────
    theo_r_km = R_e - (R_s - R_e) * (D_m / D_s)
    theo_area = math.pi * theo_r_km**2
    exp_r_px  = theo_r_km / km_pp

    # ── Fit outer circle ──────────────────────────────────
    o_cx = o_cy = o_r = 0.0
    out_inlier  = outer_pts
    out_outlier = np.empty((0, 2))
    if len(outer_pts) >= 3:
        if HAS_SKIMAGE:
            mr2, inl = ransac(outer_pts, CircleModel, 3, 3.0, max_trials=1000)
            if mr2:
                out_inlier  = outer_pts[inl] if inl.sum() > 10 else outer_pts
                out_outlier = outer_pts[~inl]
        o_cx, o_cy, o_r = fit_circle_constrained(
            out_inlier, exp_r_px, cx, cy, r_moon)

    # ── Fit inner circle ──────────────────────────────────
    i_cx = i_cy = i_r = 0.0
    inn_inlier = inner_pts
    if len(inner_pts) >= 3:
        if HAS_SKIMAGE:
            mr3, inl2 = ransac(inner_pts, CircleModel, 3, 2.0, max_trials=500)
            if mr3:
                inn_inlier = inner_pts[inl2] if inl2.sum() > 10 else inner_pts
        i_cx, i_cy, i_r = fit_circle_taubin(inn_inlier[:, 0], inn_inlier[:, 1])

    o_r_km   = o_r * km_pp if o_r else 0
    o_area   = math.pi * o_r_km**2 if o_r_km else 0
    i_r_km   = i_r * km_pp if i_r else 0
    err_r    = abs(o_r_km - theo_r_km) / theo_r_km * 100 if o_r_km else 0
    err_area = abs(o_area - theo_area) / theo_area * 100 if o_area else 0

    # ── Shadow coverage estimate ──────────────────────────
    moon_mask = np.zeros(gray.shape[:2], np.uint8)
    cv2.circle(moon_mask, (int(cx), int(cy)), int(r_moon * 0.95), 255, -1)
    mp2 = cv2.GaussianBlur(gray, (9, 9), 0)[moon_mask == 255]
    dark_pct = (float(np.sum(mp2 < np.percentile(mp2, 95) * 0.35) / len(mp2) * 100)
                if len(mp2) else 0.0)

    pfl  = min((len(p) for p in outer_profiles), default=0) if outer_profiles else 0
    pfl2 = min((len(p) for p in inner_profiles), default=0) if inner_profiles else 0

    return {
        "cx": cx, "cy": cy, "r_moon": r_moon, "km_pp": km_pp,
        "o_cx": o_cx, "o_cy": o_cy, "o_r": o_r,
        "o_r_km": o_r_km, "o_area": o_area,
        "out_inlier": out_inlier, "out_outlier": out_outlier,
        "best_outer_ray": best_outer_ray,
        "outer_mean": (np.mean([p[:pfl] for p in outer_profiles], axis=0)
                       if pfl > 10 else np.array([])),
        "i_cx": i_cx, "i_cy": i_cy, "i_r": i_r, "i_r_km": i_r_km,
        "inn_inlier": inn_inlier, "best_inner_ray": best_inner_ray,
        "inner_mean": (np.mean([p[:pfl2] for p in inner_profiles], axis=0)
                       if pfl2 > 10 else np.array([])),
        "theo_r_km": theo_r_km, "theo_area": theo_area,
        "err_r": err_r, "err_area": err_area,
        "dark_pct": dark_pct, "img_bgr": img_bgr,
    }


# ── Photometry ───────────────────────────────────────────────────────────────

def phot_background(gray8: np.ndarray, box: int = 64) -> np.ndarray:
    """
    Estimate 2-D sky background by sigma-clipped median on a coarse grid,
    then bicubic-interpolate back to full resolution.
    """
    from scipy.ndimage import zoom
    h, w = gray8.shape
    ny, nx = max(2, h // box), max(2, w // box)
    yg = np.linspace(0, h - 1, ny, dtype=int)
    xg = np.linspace(0, w - 1, nx, dtype=int)
    grid = np.zeros((ny, nx), float)
    for iy, y0 in enumerate(yg):
        for ix, x0 in enumerate(xg):
            y1, y2 = max(0, y0 - box // 2), min(h, y0 + box // 2)
            x1, x2 = max(0, x0 - box // 2), min(w, x0 + box // 2)
            patch  = gray8[y1:y2, x1:x2].astype(float)
            med    = np.median(patch)
            sigma  = 1.4826 * np.median(np.abs(patch - med))
            grid[iy, ix] = np.median(patch[patch < med + 3 * sigma])
    bkg = zoom(grid, (h / ny, w / nx), order=3)
    return bkg[:h, :w]


def phot_detect(gray8, bkg, thresh_sigma=4.0, ap_r=5,
                sky_in=8, sky_out=13, min_sep=10, max_stars=5000):
    """
    Detect stars using aperture photometry.

    Returns a dict with keys: x, y, flux, mag (instrumental), ci (concentration index).
    Returns None if no stars found.
    """
    h, w = gray8.shape
    sub  = gray8.astype(float) - bkg
    rms  = 1.4826 * np.median(np.abs(sub)) + 1e-6
    above   = sub > thresh_sigma * rms
    dilated = maximum_filter(gray8.astype(float), size=5)
    peaks   = above & (gray8.astype(float) == dilated)
    labs, n = ndlabel(peaks)
    if n == 0:
        return None

    cx_list, cy_list = [], []
    for i in range(1, n + 1):
        pts = np.argwhere(labs == i)
        cy2, cx2 = pts.mean(axis=0)
        cx_list.append(cx2)
        cy_list.append(cy2)

    xs = np.array(cx_list)
    ys = np.array(cy_list)
    if len(xs) > 1:
        keep = np.ones(len(xs), bool)
        for i in range(len(xs)):
            if not keep[i]:
                continue
            d = np.sqrt((xs - xs[i])**2 + (ys - ys[i])**2)
            keep[(d < min_sep) & (d > 0)] = False
        xs, ys = xs[keep], ys[keep]

    Y, X = np.mgrid[0:h, 0:w]
    fluxes, ci_list, vx, vy = [], [], [], []
    ap_r2 = max(2, ap_r // 2)

    for cx, cy in zip(xs, ys):
        pad = sky_out + 2
        if cx < pad or cx > w - pad or cy < pad or cy > h - pad:
            continue
        r2       = (X - cx)**2 + (Y - cy)**2
        ap_mask  = r2 <= ap_r**2
        ap2_mask = r2 <= ap_r2**2
        sky_mask = (r2 >= sky_in**2) & (r2 <= sky_out**2)
        sky_pix  = gray8[sky_mask].astype(float)
        if len(sky_pix) < 8:
            continue
        sky_med = np.median(sky_pix)
        net  = gray8[ap_mask].astype(float).sum() - sky_med * ap_mask.sum()
        net2 = gray8[ap2_mask].astype(float).sum() - sky_med * ap2_mask.sum()
        if net <= 0:
            continue
        fluxes.append(net)
        ci_list.append(net2 / net)
        vx.append(cx)
        vy.append(cy)

    if not fluxes:
        return None

    fl  = np.array(fluxes)
    ci  = np.array(ci_list)
    vx  = np.array(vx)
    vy  = np.array(vy)
    ord2 = np.argsort(fl)[::-1][:max_stars]
    fl, ci, vx, vy = fl[ord2], ci[ord2], vx[ord2], vy[ord2]

    return {
        "x":   vx,
        "y":   vy,
        "flux": fl,
        "mag":  25.0 - 2.5 * np.log10(fl),
        "ci":   ci,
    }