"""
Feature Map & Solar Eclipse Analyzer  +  CMD Plot
ครุที่ปรึกษา: วิทชภณ พวงแก้ว
Developed by: Sitthichokthq (https://github.com/Sitthichokthq) and (sitthichokthq.web.app)


"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import math
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as mgridspec
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d, maximum_filter, label as ndlabel

try:
    from skimage.measure import ransac, CircleModel
    HAS_SKIMAGE = True
except ModuleNotFoundError:
    HAS_SKIMAGE = False

# ── Astronomical Constants ──────────────────────────────
R_e              = 6371.0
R_s              = 696340.0
D_s              = 149597870.7
D_m              = 384400.0
MOON_DIAMETER_KM = 3474.8

# ── Dark Theme ──────────────────────────────────────────
T = {
    "bg":        "#0e0f14",
    "panel":     "#161820",
    "panel2":    "#1c1e28",
    "border":    "#2a2d3e",
    "fg":        "#d4d8f0",
    "fg2":       "#7880a0",
    "accent":    "#4f8fff",
    "accent2":   "#a78bfa",
    "success":   "#34d399",
    "warning":   "#fbbf24",
    "danger":    "#f87171",
    "canvas":    "#060709",
    "plot_bg":   "#0e0f14",
    "grid":      "#1e2030",
    "outer":     "#4f8fff",
    "inner":     "#f472b6",
    "button":    "#1c1e28",
    "button_fg": "#d4d8f0",
    "tab_sel":   "#252838",
}

FILTERS = ["Cross-Correlation", "Sobel Edge", "Laplacian", "Sharpen"]


# ══════════════════════════════════════════════════════
#  Image / Eclipse algorithms
# ══════════════════════════════════════════════════════

def apply_filter(gray, mode):
    kernels = {
        "Cross-Correlation": np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]),
        "Sharpen":           np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    }
    if mode in kernels:   return cv2.filter2D(gray, -1, kernels[mode])
    if mode == "Sobel Edge":
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        return cv2.magnitude(sx, sy).astype(np.uint8)
    if mode == "Laplacian": return cv2.Laplacian(gray, cv2.CV_8U)
    return gray


def fit_circle_taubin(xs, ys):
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if len(xs) < 3: raise ValueError("need >= 3 pts")
    mx, my = xs.mean(), ys.mean()
    u, v   = xs-mx, ys-my
    Suu, Svv, Suv = (u**2).sum(), (v**2).sum(), (u*v).sum()
    A = np.array([[Suu,Suv],[Suv,Svv]])
    b = np.array([0.5*(u**3+u*v**2).sum(), 0.5*(v**3+v*u**2).sum()])
    try:    uc, vc = np.linalg.solve(A, b)
    except: uc, vc = 0.0, 0.0
    return uc+mx, vc+my, float(np.sqrt(uc**2+vc**2+(Suu+Svv)/len(xs)))


def fit_circle_constrained(p, expected_r=None, cx_moon=None, cy_moon=None, r_moon=None):
    def calc_R(c): return np.sqrt((p[:,0]-c[0])**2 + (p[:,1]-c[1])**2)
    def f(c):
        residuals = calc_R(c) - c[2]
        r_pen = (c[2]-expected_r)*0.1 if expected_r else 0.0
        c_pen = 0.0
        if cx_moon and cy_moon and r_moon:
            d = np.sqrt((c[0]-cx_moon)**2+(c[1]-cy_moon)**2)
            if d < r_moon*1.5: c_pen = (r_moon*1.5-d)*10.0
        return np.append(residuals, [r_pen*len(p), c_pen*len(p)//2])
    ce  = np.mean(p, axis=0)
    gr  = expected_r if expected_r else calc_R(np.append(ce,0)).mean()
    bnd = ([-np.inf,-np.inf,expected_r*0.85],[np.inf,np.inf,expected_r*1.15]) if expected_r else (-np.inf,np.inf)
    res = least_squares(f, [ce[0],ce[1],gr], bounds=bnd)
    return res.x[0], res.x[1], res.x[2]


def correct_limb_darkening(brightness, r_vals, r_moon, u=0.8):
    r_frac    = np.clip(r_vals/r_moon, 0, 0.9999)
    cos_theta = np.sqrt(1-r_frac**2)
    corr      = 1 - u*(1-cos_theta)
    I = brightness.copy()
    vm = r_vals < 0.98*r_moon
    I[vm] = brightness[vm]/corr[vm]
    return I


def align_and_median_combine(paths):
    images, crs = [], []
    for p in paths:
        img = cv2.imread(p)
        if img is None: continue
        g = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(5,5),0)
        c = cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,100,param1=100,param2=50,minRadius=50,maxRadius=int(min(img.shape[:2])/2))
        if c is not None:
            x = np.uint16(np.around(c))[0,0]
            crs.append((x[0],x[1],x[2])); images.append(img)
    if not images: return None, None
    mr  = int(np.median([c[2] for c in crs]))
    tcx, tcy = images[0].shape[1]//2, images[0].shape[0]//2
    aligned = []
    for img,(cx,cy,r) in zip(images,crs):
        s = mr/r
        if s!=1.0:
            img=cv2.resize(img,None,fx=s,fy=s,interpolation=cv2.INTER_LANCZOS4)
            cx,cy=int(cx*s),int(cy*s)
        M=np.float32([[1,0,tcx-cx],[0,1,tcy-cy]])
        aligned.append(cv2.warpAffine(img,M,(images[0].shape[1],images[0].shape[0])))
    return np.median(np.stack(aligned,0),0).astype(np.uint8),(tcx,tcy,mr)


def detect_rings(img, cx, cy, r_moon, limb_u=0.8):
    km_pp  = MOON_DIAMETER_KM/(2*r_moon)
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur   = cv2.GaussianBlur(gray,(7,7),0)
    eq     = cv2.createCLAHE(2.0,(8,8)).apply(blur)
    angles = np.arange(0, 360, 0.5)
    max_r  = int(2.5*r_moon)
    r_vals = np.arange(0, max_r)
    outer_pts, inner_pts = [], []
    outer_profiles, inner_profiles = [], []
    best_outer_ray = best_inner_ray = None
    min_outer = min_inner = 0
    for angle in angles:
        rad = np.deg2rad(angle)
        xc  = cx + r_vals*np.cos(rad)
        yc  = cy + r_vals*np.sin(rad)
        ok  = (xc>=0)&(xc<img.shape[1])&(yc>=0)&(yc<img.shape[0])
        if not np.any(ok): continue
        xv,yv,rv = xc[ok],yc[ok],r_vals[ok]
        xi = np.clip(np.round(xv).astype(int),0,img.shape[1]-1)
        yi = np.clip(np.round(yv).astype(int),0,img.shape[0]-1)
        br = correct_limb_darkening(eq[yi,xi].astype(float), rv, r_moon, limb_u)
        wl = min(31, len(br)-(1 if len(br)%2==0 else 0))
        sm = savgol_filter(br, wl, 3) if wl>3 else br
        gd = np.gradient(sm)
        sg = cv2.GaussianBlur(gd.reshape(-1,1),(15,15),0).flatten()
        inner_zone = (rv >= r_moon*0.6) & (rv <= r_moon*1.1)
        iz = np.where(inner_zone)[0]
        if len(iz) >= 5:
            ig = sg[iz]; ii = np.argmin(ig)
            if ig[ii] < -0.3:
                bi = iz[ii]
                sw,ew = max(0,bi-5), min(len(gd),bi+6)
                if ew-sw >= 3:
                    fi  = interp1d(np.arange(sw,ew),gd[sw:ew],kind='cubic',fill_value='extrapolate')
                    xn  = np.linspace(sw,ew-1,(ew-sw)*100)
                    rfr = rv[0]+xn[np.argmin(fi(xn))]
                    inner_pts.append([cx+rfr*np.cos(rad), cy+rfr*np.sin(rad)])
                    inner_profiles.append(sm.copy())
                    if ig[ii] < min_inner:
                        min_inner = ig[ii]; best_inner_ray = (sm.copy(),gd.copy(),rv.copy(),rfr)
        mask = np.ones_like(sg, dtype=bool)
        es,ee = int(0.95*r_moon), int(1.05*r_moon)
        if ee<len(mask): mask[es:ee]=False
        mask[:int(0.15*r_moon)]=False
        vi = np.where(mask)[0]
        if len(vi) < 10: continue
        mg = sg[vi]; mi = np.argmin(mg)
        if mg[mi] > -0.5: continue
        bri = vi[mi]
        sw,ew = max(0,bri-5), min(len(gd),bri+6)
        if ew-sw < 3: continue
        fo  = interp1d(np.arange(sw,ew),gd[sw:ew],kind='cubic',fill_value='extrapolate')
        xn  = np.linspace(sw,ew-1,(ew-sw)*100)
        rfr = rv[0]+xn[np.argmin(fo(xn))]
        outer_pts.append([cx+rfr*np.cos(rad), cy+rfr*np.sin(rad)])
        outer_profiles.append(sm.copy())
        if mg[mi] < min_outer:
            min_outer = mg[mi]; best_outer_ray = (sm.copy(),gd.copy(),rv.copy(),rfr)
    outer_pts = np.array(outer_pts) if outer_pts else np.empty((0,2))
    inner_pts = np.array(inner_pts) if inner_pts else np.empty((0,2))
    theo_r_km = R_e - (R_s-R_e)*(D_m/D_s)
    theo_area = math.pi*theo_r_km**2
    exp_r_px  = theo_r_km/km_pp
    o_cx=o_cy=o_r = 0.0
    out_inlier = outer_pts; out_outlier = np.empty((0,2))
    if len(outer_pts) >= 3:
        if HAS_SKIMAGE:
            mr2, inl = ransac(outer_pts,CircleModel,3,3.0,max_trials=1000)
            if mr2:
                out_inlier  = outer_pts[inl] if inl.sum()>10 else outer_pts
                out_outlier = outer_pts[~inl]
        o_cx,o_cy,o_r = fit_circle_constrained(out_inlier,exp_r_px,cx,cy,r_moon)
    i_cx=i_cy=i_r = 0.0; inn_inlier = inner_pts
    if len(inner_pts) >= 3:
        if HAS_SKIMAGE:
            mr3, inl2 = ransac(inner_pts,CircleModel,3,2.0,max_trials=500)
            if mr3: inn_inlier = inner_pts[inl2] if inl2.sum()>10 else inner_pts
        i_cx,i_cy,i_r = fit_circle_taubin(inn_inlier[:,0], inn_inlier[:,1])
    o_r_km   = o_r*km_pp if o_r else 0
    o_area   = math.pi*o_r_km**2 if o_r_km else 0
    i_r_km   = i_r*km_pp if i_r else 0
    err_r    = abs(o_r_km-theo_r_km)/theo_r_km*100 if o_r_km else 0
    err_area = abs(o_area-theo_area)/theo_area*100 if o_area else 0
    moon_mask = np.zeros(gray.shape[:2],np.uint8)
    cv2.circle(moon_mask,(int(cx),int(cy)),int(r_moon*0.95),255,-1)
    mp2 = cv2.GaussianBlur(gray,(9,9),0)[moon_mask==255]
    dark_pct = float(np.sum(mp2<np.percentile(mp2,95)*0.35)/len(mp2)*100) if len(mp2) else 0.0
    pfl  = min((len(p) for p in outer_profiles), default=0) if outer_profiles else 0
    pfl2 = min((len(p) for p in inner_profiles), default=0) if inner_profiles else 0
    return {
        "cx":cx,"cy":cy,"r_moon":r_moon,"km_pp":km_pp,
        "o_cx":o_cx,"o_cy":o_cy,"o_r":o_r,"o_r_km":o_r_km,"o_area":o_area,
        "out_inlier":out_inlier,"out_outlier":out_outlier,
        "best_outer_ray":best_outer_ray,
        "outer_mean": np.mean([p[:pfl] for p in outer_profiles],axis=0) if pfl>10 else np.array([]),
        "i_cx":i_cx,"i_cy":i_cy,"i_r":i_r,"i_r_km":i_r_km,
        "inn_inlier":inn_inlier,"best_inner_ray":best_inner_ray,
        "inner_mean": np.mean([p[:pfl2] for p in inner_profiles],axis=0) if pfl2>10 else np.array([]),
        "theo_r_km":theo_r_km,"theo_area":theo_area,
        "err_r":err_r,"err_area":err_area,"dark_pct":dark_pct,"img_bgr":img,
    }


def dax(ax, title, xl="", yl=""):
    ax.set_facecolor(T["plot_bg"])
    ax.set_title(title, color=T["fg"], fontsize=8, fontweight="bold", pad=4)
    ax.set_xlabel(xl, color=T["fg2"], fontsize=7)
    ax.set_ylabel(yl, color=T["fg2"], fontsize=7)
    ax.tick_params(colors=T["fg2"], labelsize=6)
    for sp in ax.spines.values(): sp.set_color(T["border"])
    ax.grid(color=T["grid"], lw=0.5, ls="--", alpha=0.9)


# ══════════════════════════════════════════════════════
#  Photometry helpers  (single-image, aperture)
# ══════════════════════════════════════════════════════

def phot_background(gray8, box=64):
    from scipy.ndimage import zoom
    h, w = gray8.shape
    ny, nx = max(2, h//box), max(2, w//box)
    yg = np.linspace(0, h-1, ny, dtype=int)
    xg = np.linspace(0, w-1, nx, dtype=int)
    grid = np.zeros((ny, nx), float)
    for iy, y0 in enumerate(yg):
        for ix, x0 in enumerate(xg):
            y1,y2 = max(0,y0-box//2), min(h,y0+box//2)
            x1,x2 = max(0,x0-box//2), min(w,x0+box//2)
            patch  = gray8[y1:y2,x1:x2].astype(float)
            med    = np.median(patch)
            sigma  = 1.4826*np.median(np.abs(patch-med))
            grid[iy,ix] = np.median(patch[patch < med+3*sigma])
    bkg = zoom(grid, (h/ny, w/nx), order=3)
    return bkg[:h,:w]


def phot_detect(gray8, bkg, thresh_sigma=4.0, ap_r=5,
                sky_in=8, sky_out=13, min_sep=10, max_stars=5000):
    h, w   = gray8.shape
    sub    = gray8.astype(float) - bkg
    rms    = 1.4826 * np.median(np.abs(sub)) + 1e-6
    above  = sub > thresh_sigma * rms
    dilated = maximum_filter(gray8.astype(float), size=5)
    peaks   = above & (gray8.astype(float) == dilated)
    labs, n = ndlabel(peaks)
    if n == 0: return None
    cx_list, cy_list = [], []
    for i in range(1, n+1):
        pts = np.argwhere(labs == i)
        cy2, cx2 = pts.mean(axis=0)
        cx_list.append(cx2); cy_list.append(cy2)
    xs = np.array(cx_list); ys = np.array(cy_list)
    if len(xs) > 1:
        keep = np.ones(len(xs), bool)
        for i in range(len(xs)):
            if not keep[i]: continue
            d = np.sqrt((xs-xs[i])**2+(ys-ys[i])**2)
            keep[(d<min_sep)&(d>0)] = False
        xs, ys = xs[keep], ys[keep]
    Y, X = np.mgrid[0:h, 0:w]
    fluxes, ci_list, vx, vy = [], [], [], []
    ap_r2 = max(2, ap_r//2)
    for cx, cy in zip(xs, ys):
        pad = sky_out + 2
        if cx<pad or cx>w-pad or cy<pad or cy>h-pad: continue
        r2       = (X-cx)**2+(Y-cy)**2
        ap_mask  = r2 <= ap_r**2
        ap2_mask = r2 <= ap_r2**2
        sky_mask = (r2>=sky_in**2)&(r2<=sky_out**2)
        sky_pix  = gray8[sky_mask].astype(float)
        if len(sky_pix)<8: continue
        sky_med  = np.median(sky_pix)
        net      = gray8[ap_mask].astype(float).sum() - sky_med*ap_mask.sum()
        net2     = gray8[ap2_mask].astype(float).sum() - sky_med*ap2_mask.sum()
        if net <= 0: continue
        fluxes.append(net); ci_list.append(net2/net)
        vx.append(cx); vy.append(cy)
    if not fluxes: return None
    fl  = np.array(fluxes); ci = np.array(ci_list)
    vx  = np.array(vx);     vy = np.array(vy)
    ord2 = np.argsort(fl)[::-1][:max_stars]
    fl, ci, vx, vy = fl[ord2], ci[ord2], vx[ord2], vy[ord2]
    return {"x":vx, "y":vy, "flux":fl, "mag":25.0-2.5*np.log10(fl), "ci":ci}


# ══════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Map & Eclipse Analyzer")
        self.root.geometry("1920x1080")
        self.root.configure(bg=T["bg"])

        self.img_orig     = None
        self.img_filtered = None
        self.outer_circle = None
        self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self.eclipse_paths = []

        self.zoom   = 1.0
        self.offset = [0, 0]
        self.tool   = "measure"

        self._drawing   = False
        self._pt_start  = self._pt_end = None
        self._pan_start = (0,0)
        self._pending   = False
        self._tk_imgs   = {}

        # CMD state
        self._cmd_cat    = None
        self._cmd_thresh = tk.DoubleVar(value=4.0)
        self._cmd_ap     = tk.IntVar(value=5)
        self._cmd_tip    = tk.DoubleVar(value=0.0)
        self._cmd_style  = tk.StringVar(value="Dark")

        self._build()

    # ══════════════════════════════════════════════════
    #  Build UI
    # ══════════════════════════════════════════════════

    def _build(self):
        self._build_header()
        self._build_body()
        self._style_ttk()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=T["panel"], height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        def sep():
            tk.Frame(hdr, bg=T["border"], width=1).pack(
                side=tk.LEFT, fill=tk.Y, padx=5, pady=10)

        self._hbtn(hdr,"Open Image",   self.load_image,   "accent").pack(side=tk.LEFT,padx=(6,0),pady=10)
        sep()

        tk.Label(hdr,text="Filter",bg=T["panel"],fg=T["fg2"],font=("Helvetica",16)).pack(side=tk.LEFT,padx=(0,3))
        self.filter_var = tk.StringVar(value=FILTERS[0])
        cb = ttk.Combobox(hdr,textvariable=self.filter_var,values=FILTERS,width=14,state="readonly",font=("Helvetica",16))
        cb.pack(side=tk.LEFT,pady=12)
        cb.bind("<<ComboboxSelected>>",lambda _:self.refresh_filter())
        sep()

        self._tool_btns = {}
        for key,lbl in [("measure","Measure"),("roi","ROI"),("inspect","Inspect"),("pan","Pan")]:
            b = self._hbtn(hdr,lbl,lambda k=key:self.set_tool(k),"tool")
            b.pack(side=tk.LEFT,padx=2,pady=10)
            self._tool_btns[key] = b
        sep()

        self._hbtn(hdr,"Detect Circles",self.detect_circles,"success").pack(side=tk.LEFT,padx=3,pady=10)
        self._hbtn(hdr,"Sobel Fit",      self.sobel_fit,     "tool"  ).pack(side=tk.LEFT,padx=2,pady=10)
        sep()

        self._hbtn(hdr,"Analyze Eclipse",self.eclipse_analyze,"warning").pack(side=tk.LEFT,padx=3,pady=10)
        tk.Label(hdr,text="u",bg=T["panel"],fg=T["fg2"],font=("Helvetica",16)).pack(side=tk.LEFT,padx=(6,1))
        self.limb_u = tk.Scale(hdr,from_=0,to=100,orient=tk.HORIZONTAL,length=60,bg=T["panel"],fg=T["fg"], highlightthickness=0,troughcolor=T["border"], font=("Helvetica",7),bd=0,sliderlength=12)
        self.limb_u.set(80)
        self.limb_u.pack(side=tk.LEFT,pady=10)
        sep()

        # CMD controls inline in header
        tk.Label(hdr,text="σ:",bg=T["panel"],fg=T["fg2"],font=("Helvetica",16)).pack(side=tk.LEFT,padx=(0,1))
        tk.Spinbox(hdr,from_=1,to=20,increment=0.5,format="%.1f",textvariable=self._cmd_thresh,width=4,font=("Menlo",8),bg=T["panel2"],fg=T["fg"],buttonbackground=T["panel2"],relief=tk.FLAT,bd=0,highlightthickness=1,highlightbackground=T["border"]).pack(side=tk.LEFT,pady=12)
        tk.Label(hdr,text="ap:",bg=T["panel"],fg=T["fg2"],font=("Helvetica",16)).pack(side=tk.LEFT,padx=(4,1))
        tk.Spinbox(hdr,from_=2,to=20,textvariable=self._cmd_ap,width=3,font=("Menlo",8),bg=T["panel2"],fg=T["fg"],buttonbackground=T["panel2"],relief=tk.FLAT,bd=0,highlightthickness=1,highlightbackground=T["border"]).pack(side=tk.LEFT,pady=12)
        self._hbtn(hdr,"▶ Detect Stars",self.run_cmd,"accent2").pack(side=tk.LEFT,padx=4,pady=10)
        sep()

        self.lbl_status = tk.Label(hdr,text="No image loaded",bg=T["panel"],fg=T["fg2"],font=("Helvetica",8))
        self.lbl_status.pack(side=tk.LEFT,padx=6)
        self._hbtn(hdr,"Clear",self._clear_all,"danger").pack(side=tk.RIGHT,padx=8,pady=10)

    def _build_body(self):
        body = tk.Frame(self.root,bg=T["bg"])
        body.pack(fill=tk.BOTH,expand=True,padx=5,pady=(2,0))
        body.columnconfigure(0,weight=3)
        body.columnconfigure(1,weight=3)
        body.columnconfigure(2,weight=2)
        body.rowconfigure(0,weight=1)

        # Left: Original
        lf_l = self._lframe(body,"Original")
        lf_l.grid(row=0,column=0,sticky="nsew",padx=(0,3))
        self.cv_orig = tk.Canvas(lf_l,bg=T["canvas"],highlightthickness=0)
        self.cv_orig.pack(fill=tk.BOTH,expand=True,padx=2,pady=2)

        # Mid: Filtered
        lf_m = self._lframe(body,"Filtered")
        lf_m.grid(row=0,column=1,sticky="nsew",padx=3)
        self.cv_filt = tk.Canvas(lf_m,bg=T["canvas"],highlightthickness=0)
        self.cv_filt.pack(fill=tk.BOTH,expand=True,padx=2,pady=2)
        self._lbl_filter = lf_m.winfo_children()[0]

        # Right: Analysis text + Notebook(Profiles | CMD)
        right = tk.Frame(body,bg=T["bg"])
        right.grid(row=0,column=2,sticky="nsew",padx=(3,0))
        right.rowconfigure(0,weight=1)
        right.rowconfigure(1,weight=2)
        right.columnconfigure(0,weight=1)

        lf_info = self._lframe(right,"Analysis")
        lf_info.grid(row=0,column=0,sticky="nsew",pady=(0,3))
        self.info_box = tk.Text(lf_info,font=("Menlo",8),state=tk.DISABLED,
                                wrap=tk.WORD,bg=T["panel2"],fg=T["fg"],
                                relief=tk.FLAT,bd=0,padx=8,pady=6)
        self.info_box.pack(fill=tk.BOTH,expand=True,padx=2,pady=2)

        # Notebook
        self._nb = ttk.Notebook(right)
        self._nb.grid(row=1,column=0,sticky="nsew")

        tab_prof = tk.Frame(self._nb,bg=T["panel"])
        tab_cmd  = tk.Frame(self._nb,bg=T["panel"])
        self._nb.add(tab_prof,text="  Profiles  ")
        self._nb.add(tab_cmd, text="  CMD Plot  ")

        self._build_profiles_tab(tab_prof)
        self._build_cmd_tab(tab_cmd)

        # Canvas bindings
        self.cv_filt.bind("<ButtonPress-1>",  self._on_press)
        self.cv_filt.bind("<B1-Motion>",       self._on_drag)
        self.cv_filt.bind("<ButtonRelease-1>",self._on_release)
        self.cv_filt.bind("<Motion>",          self._on_hover)
        for cv in (self.cv_filt,self.cv_orig):
            cv.bind("<MouseWheel>",self._on_zoom)

        sb = tk.Frame(self.root,bg=T["button"],height=20)
        sb.pack(fill=tk.X)
        sb.pack_propagate(False)
        self.lbl_cursor = tk.Label(sb,text="  x=—  y=—",anchor="w",bg=T["button"],fg=T["button_fg"],font=("Menlo",7))
        self.lbl_cursor.pack(side=tk.LEFT)
        self.lbl_zoom = tk.Label(sb,text="1.00x  ",anchor="e",bg=T["button"],fg=T["button_fg"],font=("Menlo",7))
        self.lbl_zoom.pack(side=tk.RIGHT)

    # ── Profiles tab ──────────────────────────────────

    def _build_profiles_tab(self, parent):
        self._fig_prof = Figure(figsize=(5,5.5),facecolor=T["plot_bg"])
        self._fig_prof.subplots_adjust(hspace=0.6,wspace=0.4,top=0.92,bottom=0.08,left=0.13,right=0.97)
        self._ax_outer  = self._fig_prof.add_subplot(2,2,1)
        self._ax_inner  = self._fig_prof.add_subplot(2,2,2)
        self._ax_radial = self._fig_prof.add_subplot(2,2,3)
        self._ax_hist   = self._fig_prof.add_subplot(2,2,4)
        for ax,t in [(self._ax_outer,"Outer Edge"),(self._ax_inner,"Inner Edge"),(self._ax_radial,"Radial Profile"),(self._ax_hist,"Histogram")]:
            dax(ax,t)
        self._cv_prof = FigureCanvasTkAgg(self._fig_prof,master=parent)
        self._cv_prof.draw()
        self._cv_prof.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=2,pady=2)

    # ── CMD tab ───────────────────────────────────────

    def _build_cmd_tab(self, parent):
        # param mini-bar
        bar = tk.Frame(parent,bg=T["panel2"],height=28)
        bar.pack(fill=tk.X); bar.pack_propagate(False)

        def lbl(t):
            tk.Label(bar,text=t,bg=T["panel2"],fg=T["fg2"],font=("Helvetica",7)).pack(side=tk.LEFT,padx=(6,1),pady=5)

        lbl("Tip mT:")
        tk.Spinbox(bar,from_=-30,to=30,increment=0.01,format="%.3f",textvariable=self._cmd_tip,width=7,font=("Menlo",7),bg=T["panel"],fg=T["fg"],buttonbackground=T["panel"],relief=tk.FLAT,bd=0,highlightthickness=1,highlightbackground=T["border"]).pack(side=tk.LEFT,pady=3)
        self._hbtn(bar,"Set",self._cmd_redraw,"warning").pack(side=tk.LEFT,padx=3,pady=3)
        tk.Frame(bar,bg=T["border"],width=1).pack(side=tk.LEFT,fill=tk.Y,padx=5,pady=3)
        lbl("Style:")
        cb2 = ttk.Combobox(bar,textvariable=self._cmd_style,state="readonly",values=["Dark","White"],width=7,font=("Helvetica",7))
        cb2.pack(side=tk.LEFT,pady=3)
        cb2.bind("<<ComboboxSelected>>",lambda _:self._cmd_redraw())
        self._hbtn(bar,"Save PNG",self._cmd_save,"tool").pack(side=tk.RIGHT,padx=6,pady=3)

        # 3-panel figure
        self._fig_cmd = Figure(figsize=(5,5.2),facecolor=T["plot_bg"])
        gs = mgridspec.GridSpec(1,3,figure=self._fig_cmd,
                                width_ratios=[3,1.2,1.2],
                                left=0.09,right=0.97,
                                top=0.90,bottom=0.11,wspace=0.04)
        self._ax_c1 = self._fig_cmd.add_subplot(gs[0])
        self._ax_c2 = self._fig_cmd.add_subplot(gs[1],sharey=self._ax_c1)
        self._ax_c3 = self._fig_cmd.add_subplot(gs[2],sharey=self._ax_c1)
        for ax in (self._ax_c1,self._ax_c2,self._ax_c3):
            ax.set_facecolor(T["plot_bg"])
            for sp in ax.spines.values(): sp.set_color(T["border"])
            ax.tick_params(colors=T["fg2"],labelsize=7)

        self._cv_cmd = FigureCanvasTkAgg(self._fig_cmd,master=parent)
        self._cv_cmd.draw()
        self._cv_cmd.get_tk_widget().pack(fill=tk.BOTH,expand=True,padx=2,pady=(0,2))
        self._cv_cmd.mpl_connect("motion_notify_event",self._cmd_hover)

        sb = tk.Frame(parent,bg=T["button"],height=18)
        sb.pack(fill=tk.X); sb.pack_propagate(False)
        self._lbl_cmd_info = tk.Label(sb,text="  Upload image  ▶ Detect Stars",bg=T["button"],fg=T["fg2"],font=("Menlo",7),anchor="w")
        self._lbl_cmd_info.pack(side=tk.LEFT)

        self._cmd_welcome()

    def _style_ttk(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TCombobox",fieldbackground=T["panel2"],background=T["panel2"],
                    foreground=T["fg"],selectbackground=T["panel2"],
                    selectforeground=T["fg"],arrowcolor=T["fg2"])
        s.configure("TNotebook",background=T["bg"],borderwidth=0,tabmargins=0)
        s.configure("TNotebook.Tab",background=T["panel"],foreground=T["fg2"],
                    padding=(10,4),font=("Helvetica",9),borderwidth=0)
        s.map("TNotebook.Tab",background=[("selected",T["tab_sel"])],foreground=[("selected",T["fg"])])
        self._hi_tool("measure")

    # ══════════════════════════════════════════════════
    #  Widget helper
    # ══════════════════════════════════════════════════

    def _lframe(self,parent,title):
        f = tk.Frame(parent,bg=T["panel"],highlightbackground=T["border"],highlightthickness=1)
        tk.Label(f,text=title.upper(),bg=T["panel"],fg=T["fg2"],font=("TH Sarabun New",16,"bold"),padx=6,pady=3).pack(anchor="w")
        return f

    def _hbtn(self,parent,text,cmd,style="tool"):
        c = {"accent":(T["accent"],T["bg"]),"accent2":(T["accent2"],T["bg"]),"success":(T["success"],T["bg"]),"warning":(T["warning"],T["bg"]),"danger":(T["danger"],T["bg"]),"tool":(T["panel2"],T["fg"])}
        bg,fg = c.get(style,(T["panel2"],T["fg"]))
        return tk.Button(parent,text=text,command=cmd,bg=bg,fg=fg,activebackground=T["border"],activeforeground=T["fg"],font=("Helvetica",9),relief=tk.FLAT,cursor="hand2", padx=9,pady=3,bd=0,highlightthickness=0)

    # ══════════════════════════════════════════════════
    #  Load / Filter
    # ══════════════════════════════════════════════════

    def load_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("All","*.*")])
        if not p: return
        raw = cv2.imread(p)
        if raw is None: messagebox.showerror("Error","Cannot open file"); return
        self.img_orig = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        self._reset(); self.refresh_filter()

    def eclipse_open(self):
        paths = filedialog.askopenfilenames(
            title="Open eclipse image(s)",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("All","*.*")])
        if not paths: return
        self.eclipse_paths = list(paths)
        raw = cv2.imread(paths[0])
        if raw is None: return
        self.img_orig = cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)
        self._reset(); self.refresh_filter()
        self._status(f"{len(paths)} frame(s) ready",T["warning"])

    def refresh_filter(self):
        if self.img_orig is None: return
        gray = cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2GRAY)
        self.img_filtered = apply_filter(gray,self.filter_var.get())
        h,w = self.img_filtered.shape
        self._lbl_filter.config(text=f"FILTERED  {self.filter_var.get()}  {w}×{h}")
        self._cmd_cat = None
        self._cmd_welcome()
        self._redraw()

    def _reset(self):
        self.outer_circle = self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self.zoom, self.offset = 1.0,[0,0]
        self._cmd_cat = None
        self._clear_profiles()
        self._cmd_welcome()
        self._status("Image loaded",T["fg2"])

    # ══════════════════════════════════════════════════
    #  Circle detection
    # ══════════════════════════════════════════════════

    def detect_circles(self):
        if self.img_orig is None:
            messagebox.showwarning("","Open an image first"); return
        gray = cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(9,9),2); h,w=gray.shape
        circles = cv2.HoughCircles(blur,cv2.HOUGH_GRADIENT,dp=1.2,minDist=max(20,min(h,w)//8),param1=60,param2=30,minRadius=max(8,min(h,w)//30),maxRadius=min(h,w)//2)
        if circles is None: messagebox.showinfo("","No circles detected"); return
        circles = np.round(circles[0]).astype(int)
        circles = circles[circles[:,2].argsort()[::-1]]
        self.outer_circle = tuple(circles[0])
        self.inner_circle = tuple(circles[1]) if len(circles)>=2 else None
        self._log_circles(); self._update_basic_plots(); self._redraw()
        self._status(f"Outer r={circles[0][2]}px"+ (f"  Inner r={circles[1][2]}px" if len(circles)>=2 else ""),T["success"])

    def sobel_fit(self):
        if self.img_orig is None: return
        gray = cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),1.5)
        sx = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
        sy = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
        mag = np.sqrt(sx**2+sy**2)
        for thr,hi,attr in [(np.percentile(mag,97),None,"outer_circle"),(np.percentile(mag,94),np.percentile(mag,97),"inner_circle")]:
            mask=((mag>=thr) if hi is None else ((mag>=thr)&(mag<hi))).astype(np.uint8)
            ys2,xs2=np.where(mask>0)
            if len(xs2)<6: continue
            try:
                cx2,cy2,r2=fit_circle_taubin(xs2,ys2)
                setattr(self,attr,(int(round(cx2)),int(round(cy2)),int(round(r2))))
            except: pass
        self._log_circles(); self._update_basic_plots(); self._redraw()

    # ══════════════════════════════════════════════════
    #  Eclipse analysis
    # ══════════════════════════════════════════════════

    def eclipse_analyze(self):
        if not self.eclipse_paths and self.img_orig is None:
            messagebox.showwarning("","Open eclipse image first"); return
        self._status("Analyzing…",T["warning"]); self.root.update()
        try:
            u = self.limb_u.get()/100.0
            if len(self.eclipse_paths)>1:
                img,info=align_and_median_combine(self.eclipse_paths)
                if img is None: raise RuntimeError("Alignment failed")
                cx,cy,r_moon=info
            else:
                img=(cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2BGR)if self.img_orig is not None else cv2.imread(self.eclipse_paths[0]))
                if img is None: raise RuntimeError("Cannot load image")
                g=cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(5,5),0)
                c=cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,100,param1=100,param2=50,minRadius=50,maxRadius=int(min(img.shape[:2])/2))
                if c is None: raise RuntimeError("Moon not detected")
                x=np.uint16(np.around(c))[0,0]
                cx,cy,r_moon=int(x[0]),int(x[1]),int(x[2])
            res=detect_rings(img,cx,cy,r_moon,u)
            if res is None: raise RuntimeError("No umbra edge found")
            self.eclipse=res
            if res["o_r"]>0: self.outer_circle=(int(res["o_cx"]),int(res["o_cy"]),int(res["o_r"]))
            if res["i_r"]>0: self.inner_circle=(int(res["i_cx"]),int(res["i_cy"]),int(res["i_r"]))
            self._log_eclipse(); self._update_eclipse_plots(); self._redraw()
            self._status(f"Outer {res['o_r_km']:,.0f} km  Inner {res['i_r_km']:,.0f} km"f"  err {res['err_r']:.2f}%",T["success"])
        except Exception as e:
            messagebox.showerror("Error",str(e)); self._status("Analysis failed",T["danger"])

    # ══════════════════════════════════════════════════
    #  CMD — ตรวจหาดาวจากภาพ filtered แล้ว plot
    # ══════════════════════════════════════════════════

    def run_cmd(self):
        if self.img_filtered is None:
            messagebox.showwarning("","Upload image  ▶ Detect Stars"); return
        self._status("Detecting stars from filtered image…",T["accent2"]); self.root.update()
        try:
            gray = self.img_filtered
            bkg  = phot_background(gray, box=max(32,min(gray.shape)//16))
            cat  = phot_detect(gray,bkg,thresh_sigma=self._cmd_thresh.get(),ap_r=self._cmd_ap.get())
            if cat is None:
                raise RuntimeError("ตรวจไม่พบดาว — ลด threshold σ หรือเพิ่ม aperture")
            self._cmd_cat = cat
            self._cmd_tip.set(round(float(np.percentile(cat["mag"],3)),3))
            self._cmd_draw()
            self._nb.select(1)
            self._status(f"Found {len(cat['x']):,} stars from filtered image",T["success"])
            self._lbl_cmd_info.config(
                text=(f"  N={len(cat['x']):,} stars  |  "f"filter={self.filter_var.get()}  σ={self._cmd_thresh.get():.1f}  "f"ap={self._cmd_ap.get()}px"),
                fg=T["success"])
            self._redraw()   # show star markers on filtered canvas
        except Exception as e:
            messagebox.showerror("Detection Error",str(e))
            self._status("Detection failed",T["danger"])

    def _cmd_welcome(self):
        for ax in (self._ax_c1,self._ax_c2,self._ax_c3):
            ax.cla(); ax.set_facecolor(T["plot_bg"])
            for sp in ax.spines.values(): sp.set_color(T["border"])
            ax.set_xticks([]); ax.set_yticks([])
        self._ax_c1.text(0.5,0.5,
            "Upload image  ▶ Detect Stars\n\n"
            "Graph will use the Filtered image\nthat is selected on the main page\n\n"
            "X = Concentration Index (CI)\n"
            "Y = Instrumental Magnitude",
            transform=self._ax_c1.transAxes,ha="center",va="center",
            color=T["fg2"],fontsize=10,
            bbox=dict(boxstyle="round,pad=0.8",fc=T["panel2"],ec=T["border"],alpha=0.9))
        self._fig_cmd.suptitle("CMD — Color-Magnitude Diagram",color=T["fg"],fontsize=9,fontweight="bold")
        self._cv_cmd.draw_idle()

    def _cmd_redraw(self):
        if self._cmd_cat is not None: self._cmd_draw()

    def _cmd_draw(self):
        cat = self._cmd_cat
        if cat is None: return

        pub    = self._cmd_style.get()=="White"
        bg_c   = "white"    if pub else T["plot_bg"]
        fg_c   = "black"    if pub else T["fg"]
        tick_c = "black"    if pub else T["fg2"]
        grid_c = "#cccccc"  if pub else T["grid"]
        sp_c   = "black"    if pub else T["border"]
        red_c  = "darkred"  if pub else "#ef4444"
        gray_c = "gray"     if pub else "#9ca3af"

        self._fig_cmd.set_facecolor(bg_c)
        for ax in (self._ax_c1,self._ax_c2,self._ax_c3):
            ax.cla(); ax.set_facecolor(bg_c)
            for sp in ax.spines.values(): sp.set_color(sp_c)
            ax.tick_params(colors=tick_c,labelsize=7)
            ax.grid(color=grid_c,lw=0.4,ls="--",alpha=0.5)

        mag = cat["mag"]; ci = cat["ci"]
        tip = self._cmd_tip.get()

        # ── Scatter CMD: CI vs mag ─────────────────────
        ax = self._ax_c1
        try:
            from scipy.stats import gaussian_kde
            rng  = np.random.default_rng(0)
            n    = min(5000,len(mag))
            idx  = rng.choice(len(mag),size=n,replace=False)
            kde  = gaussian_kde(np.vstack([ci[idx],mag[idx]]),bw_method=0.07)
            dens = kde(np.vstack([ci[idx],mag[idx]]))
        except Exception:
            idx  = np.arange(min(5000,len(mag)))
            dens = np.ones(len(idx))

        bg_mask = np.ones(len(mag),bool); bg_mask[idx]=False
        ax.scatter(ci[bg_mask],mag[bg_mask],s=0.4,c="#333355" if not pub else "#d0d0d0",alpha=0.25,rasterized=True)
        ax.scatter(ci[idx],mag[idx],s=1.0,c=dens,cmap="inferno" if not pub else "Blues",alpha=0.85,rasterized=True,linewidths=0)

        if tip!=0:
            ax.axhline(tip,color=gray_c,lw=1.4)
            Ir=np.percentile(mag,[1,99])
            ax.text(0.02,tip-(Ir[1]-Ir[0])*0.015,f"mT = {tip:.3f}",
                    color=fg_c,fontsize=9,fontweight="bold",
                    transform=ax.get_yaxis_transform())

        Ip=np.percentile(mag,[1,99])
        ax.set_ylim(Ip[1]+0.4,Ip[0]-0.4)
        ax.set_xlim(-0.05,1.1)
        ax.set_xlabel("Concentration Index (CI)",color=fg_c,fontsize=8)
        ax.set_ylabel("Instrumental Mag",        color=fg_c,fontsize=8)
        ax.text(0.02,0.99,f"N = {len(mag):,}",
                transform=ax.transAxes,color=fg_c,fontsize=9,fontweight="bold",va="top")
        fname_lbl = self.filter_var.get()
        ax.set_title(fname_lbl,color=fg_c,fontsize=7,pad=3)

        # ── CI profile per mag bin ─────────────────────
        ax2 = self._ax_c2
        n_bins=max(20,len(mag)//150)
        bins_m=np.linspace(Ip[0]-0.2,Ip[1]+0.2,n_bins)
        mid_m=(bins_m[:-1]+bins_m[1:])/2
        ci_med=np.full(len(mid_m),np.nan)
        ci_p16=np.full(len(mid_m),np.nan); ci_p84=np.full(len(mid_m),np.nan)
        for k,(lo,hi) in enumerate(zip(bins_m[:-1],bins_m[1:])):
            sel=(mag>=lo)&(mag<hi)
            if sel.sum()>4:
                ci_med[k]=np.median(ci[sel])
                ci_p16[k]=np.percentile(ci[sel],16)
                ci_p84[k]=np.percentile(ci[sel],84)
        ok=~np.isnan(ci_med)
        for i in np.where(ok)[0]:
            ax2.plot([ci_p16[i],ci_p84[i]],[mid_m[i],mid_m[i]],color="#444466" if not pub else "#aaaaaa",lw=0.8,alpha=0.5)
        ax2.plot(ci_med[ok],mid_m[ok],color=red_c,lw=2.2)
        if tip!=0: ax2.axhline(tip,color=gray_c,lw=1.4)
        ax2.set_xlabel("CI",color=fg_c,fontsize=8)
        ax2.set_title("CI Profile",color=fg_c,fontsize=7,pad=3)
        ax2.yaxis.set_tick_params(labelleft=False)
        ax2.spines["left"].set_visible(False)

        # ── Luminosity Function ────────────────────────
        ax3 = self._ax_c3
        counts,edges=np.histogram(mag,bins=max(30,len(mag)//100))
        mid_lf=(edges[:-1]+edges[1:])/2
        sm=gaussian_filter1d(counts.astype(float),sigma=1.5)
        ax3.plot(sm,mid_lf,color=fg_c,lw=1.6,drawstyle="steps-mid")
        ax3.fill_betweenx(mid_lf,0,sm,alpha=0.18,color="black" if pub else T["fg"])
        if tip!=0: ax3.axhline(tip,color=gray_c,lw=1.4)
        ax3.invert_xaxis()
        ax3.set_xlabel("N",color=fg_c,fontsize=8)
        ax3.set_title("Lum. Function",color=fg_c,fontsize=7,pad=3)
        ax3.yaxis.set_tick_params(labelleft=False)
        ax3.spines["left"].set_visible(False)

        self._fig_cmd.suptitle(
            f"CMD  —  {fname_lbl}  —  N={len(mag):,} stars",
            color=fg_c,fontsize=9,fontweight="bold",y=0.98)
        self._cv_cmd.draw_idle()

    def _cmd_hover(self,event):
        if event.inaxes is self._ax_c1 and event.xdata and event.ydata:
            self._lbl_cmd_info.config(
                text=f"  CI={event.xdata:.3f}  mag={event.ydata:.3f}")

    def _cmd_save(self):
        path=filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG","*.png"),("PDF","*.pdf"),("SVG","*.svg")])
        if not path: return
        pub=self._cmd_style.get()=="White"
        self._fig_cmd.savefig(path,dpi=200 if pub else 150,facecolor=self._fig_cmd.get_facecolor(),bbox_inches="tight")
        self._lbl_cmd_info.config(text=f"  Saved: {path.split('/')[-1]}",fg=T["success"])

    # ══════════════════════════════════════════════════
    #  Profile plots
    # ══════════════════════════════════════════════════

    def _clear_profiles(self):
        for ax,t in [(self._ax_outer,"Outer Edge"),(self._ax_inner,"Inner Edge"),(self._ax_radial,"Radial Profile"),(self._ax_hist,"Histogram")]:
            ax.cla(); dax(ax,t)
        self._cv_prof.draw_idle()

    def _radial_profile_for(self,circle,gray):
        cx,cy,r=circle; h,w=gray.shape; sw=int(r*0.45)
        dr=np.arange(-sw,sw+1,dtype=float)
        profiles=np.full((180,len(dr)),np.nan)
        for i in range(180):
            ca=math.cos(math.pi*i/180); sa=math.sin(math.pi*i/180)
            for j,d in enumerate(dr):
                px,py=int(round(cx+(r+d)*ca)),int(round(cy+(r+d)*sa))
                if 0<=px<w and 0<=py<h: profiles[i,j]=gray[py,px]
        mp=np.nanmean(profiles,axis=0); sp=np.nanstd(profiles,axis=0)
        return dr,mp,sp,np.gradient(mp)

    def _update_basic_plots(self):
        if self.img_orig is None: return
        gray=cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2GRAY)
        for circle,ax,col,lbl in [
            (self.outer_circle,self._ax_outer,T["outer"],"Outer"),
            (self.inner_circle,self._ax_inner,T["inner"],"Inner")]:
            ax.cla(); dax(ax,f"{lbl} Edge","dr (px)","I")
            if circle is None: continue
            dr,mp,sp,gp=self._radial_profile_for(circle,gray)
            ei=int(np.argmin(gp))
            ax.plot(dr,mp,color=col,lw=1.4,label="Mean")
            ax.fill_between(dr,mp-sp,mp+sp,alpha=0.12,color=col)
            ax.plot(dr,gp*3,color=T["fg2"],lw=0.9,ls="--",label="Grad×3")
            ax.axvline(dr[ei],color=T["warning"],lw=1.2,ls=":",label=f"{dr[ei]:+.1f}px")
            ax.legend(fontsize=6,labelcolor=T["fg"],loc="upper right")
        self._ax_radial.cla(); dax(self._ax_radial,"Radial Profile","r (px)","I")
        for circle,col,lbl in [(self.outer_circle,T["outer"],"outer"),
                                (self.inner_circle,T["inner"],"inner")]:
            if circle is None: continue
            cx,cy,r=circle; rv=np.arange(0,int(r*1.5)); profs=[]
            for i in range(0,360,3):
                ca,sa=math.cos(math.pi*i/180),math.sin(math.pi*i/180)
                xi=np.clip((cx+rv*ca).astype(int),0,gray.shape[1]-1)
                yi=np.clip((cy+rv*sa).astype(int),0,gray.shape[0]-1)
                profs.append(gray[yi,xi].astype(float))
            mp2=np.mean(profs,axis=0)
            self._ax_radial.plot(rv,mp2,color=col,lw=1.3,label=lbl)
            self._ax_radial.axvline(r,color=col,lw=1,ls="--",alpha=0.5)
        self._ax_radial.legend(fontsize=6,labelcolor=T["fg"])
        self._ax_hist.cla(); dax(self._ax_hist,"Histogram","I","n")
        if self.img_filtered is not None:
            self._ax_hist.hist(self.img_filtered.flatten(),bins=64,range=(0,256),color=T["accent2"],alpha=0.8,edgecolor="none")
        self._cv_prof.draw_idle()

    def _update_eclipse_plots(self):
        r=self.eclipse
        if r is None: return
        gray=cv2.cvtColor(r["img_bgr"],cv2.COLOR_BGR2GRAY)
        self._ax_outer.cla(); dax(self._ax_outer,"Outer Edge (Umbra)","dr (px)","I")
        if self.outer_circle:
            dr,mp,sp,gp=self._radial_profile_for(self.outer_circle,gray); ei=int(np.argmin(gp))
            self._ax_outer.plot(dr,mp,color=T["outer"],lw=1.4)
            self._ax_outer.fill_between(dr,mp-sp,mp+sp,alpha=0.12,color=T["outer"])
            self._ax_outer.plot(dr,gp*3,color=T["fg2"],lw=0.9,ls="--")
            self._ax_outer.axvline(dr[ei],color=T["warning"],lw=1.3,ls=":",label=f"edge {dr[ei]:+.1f}px")
            self._ax_outer.legend(fontsize=6,labelcolor=T["fg"])
        self._ax_inner.cla(); dax(self._ax_inner,"Inner Edge (Moon limb)","dr (px)","I")
        if self.inner_circle:
            dr2,mp2,sp2,gp2=self._radial_profile_for(self.inner_circle,gray); ei2=int(np.argmin(gp2))
            self._ax_inner.plot(dr2,mp2,color=T["inner"],lw=1.4)
            self._ax_inner.fill_between(dr2,mp2-sp2,mp2+sp2,alpha=0.12,color=T["inner"])
            self._ax_inner.plot(dr2,gp2*3,color=T["fg2"],lw=0.9,ls="--")
            self._ax_inner.axvline(dr2[ei2],color=T["warning"],lw=1.3,ls=":",label=f"edge {dr2[ei2]:+.1f}px")
            self._ax_inner.legend(fontsize=6,labelcolor=T["fg"])
        self._ax_radial.cla(); dax(self._ax_radial,"Full Radial","r (px)","I")
        cx,cy,rm=r["cx"],r["cy"],r["r_moon"]; rv2=np.arange(0,int(1.8*rm)); profs=[]
        for i in range(0,360,3):
            ca,sa=math.cos(math.pi*i/180),math.sin(math.pi*i/180)
            xi=np.clip((cx+rv2*ca).astype(int),0,gray.shape[1]-1)
            yi=np.clip((cy+rv2*sa).astype(int),0,gray.shape[0]-1)
            profs.append(gray[yi,xi].astype(float))
        mp3=np.mean(profs,axis=0); sp3=np.std(profs,axis=0)
        self._ax_radial.plot(rv2,mp3,color=T["accent"],lw=1.4)
        self._ax_radial.fill_between(rv2,mp3-sp3,mp3+sp3,alpha=0.1,color=T["accent"])
        self._ax_radial.axvline(rm,color=T["inner"],lw=1.1,ls="--",label=f"Moon r={rm}px")
        if r["o_r"]>0:
            self._ax_radial.axvline(r["o_r"],color=T["outer"],lw=1.1,ls="--",label=f"Umbra r={r['o_r']:.0f}px")
        self._ax_radial.legend(fontsize=6,labelcolor=T["fg"])
        self._ax_hist.cla(); dax(self._ax_hist,"Histogram","I","n")
        gf=gray.flatten()
        self._ax_hist.hist(gf,bins=64,range=(0,256),color=T["accent2"],alpha=0.8,edgecolor="none")
        self._ax_hist.axvline(np.percentile(gf,95)*0.35,color=T["danger"],lw=1,ls="--",label="shadow")
        self._ax_hist.legend(fontsize=6,labelcolor=T["fg"])
        self._cv_prof.draw_idle()

    # ══════════════════════════════════════════════════
    #  Canvas render
    # ══════════════════════════════════════════════════

    def _redraw(self):
        if not self._pending:
            self._pending=True; self.root.after(16,self._do_redraw)

    def _do_redraw(self):
        self._pending=False
        if self.img_filtered is None: return
        self._draw_canvas(self.cv_orig,self._make_orig())
        self._draw_canvas(self.cv_filt,self._make_filt())

    def _scale(self,canvas,iw,ih):
        cw=canvas.winfo_width() or 500; ch=canvas.winfo_height() or 400
        sc=min(cw/iw,ch/ih)*0.92*self.zoom
        return sc,cw//2+self.offset[0]-int(iw*sc)//2,ch//2+self.offset[1]-int(ih*sc)//2

    def _make_orig(self):
        h,w=self.img_orig.shape[:2]; sc,ox,oy=self._scale(self.cv_orig,w,h)
        nw,nh=max(1,int(w*sc)),max(1,int(h*sc))
        pil=Image.fromarray(self.img_orig).resize((nw,nh),Image.BILINEAR)
        self._draw_rings(ImageDraw.Draw(pil),sc); return pil

    def _make_filt(self):
        h,w=self.img_filtered.shape; sc,ox,oy=self._scale(self.cv_filt,w,h)
        nw,nh=max(1,int(w*sc)),max(1,int(h*sc))
        pil=Image.fromarray(self.img_filtered).resize((nw,nh),Image.BILINEAR)
        draw=ImageDraw.Draw(pil)
        # star markers overlay
        if self._cmd_cat is not None:
            for sx,sy in zip(self._cmd_cat["x"],self._cmd_cat["y"]):
                px,py=int(sx*sc),int(sy*sc)
                draw.ellipse([px-3,py-3,px+3,py+3],outline=T["accent2"],width=1)
        if self.measure:
            p1,p2,d=self.measure
            s1=(int(p1[0]*sc),int(p1[1]*sc)); s2=(int(p2[0]*sc),int(p2[1]*sc))
            draw.line([s1,s2],fill="#ffffff",width=2)
            for pt in (s1,s2): draw.ellipse([pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4],fill=T["danger"])
            km=f"  {d*self.eclipse['km_pp']:.0f}km" if self.eclipse else ""
            draw.text(((s1[0]+s2[0])//2+4,(s1[1]+s2[1])//2-12),f"{d:.1f}px{km}",fill=T["warning"])
        if self.roi:
            x1,y1,x2,y2=self.roi
            draw.rectangle([int(x1*sc),int(y1*sc),int(x2*sc),int(y2*sc)],outline=T["success"],width=2)
        self._draw_rings(draw,sc)
        if self._drawing and self._pt_start and self._pt_end:
            s1=(int(self._pt_start[0]*sc),int(self._pt_start[1]*sc))
            s2=(int(self._pt_end[0]*sc),  int(self._pt_end[1]*sc))
            if self.tool=="roi":       draw.rectangle([s1,s2],outline=T["warning"],width=1)
            elif self.tool=="measure": draw.line([s1,s2],fill=T["accent"],width=1)
        return pil

    def _draw_rings(self,draw,sc):
        for circle,col,lbl in [(self.outer_circle,T["outer"],"outer"),
                                (self.inner_circle,T["inner"],"inner")]:
            if circle is None: continue
            cx,cy,r=circle; scx,scy,sr=int(cx*sc),int(cy*sc),int(r*sc)
            pts=[(scx+int(sr*math.cos(2*math.pi*i/360)),scy+int(sr*math.sin(2*math.pi*i/360))) for i in range(361)]
            draw.line(pts,fill=col,width=2)
            draw.line([(scx-8,scy),(scx+8,scy)],fill=col,width=1)
            draw.line([(scx,scy-8),(scx,scy+8)],fill=col,width=1)
            draw.text((scx+sr+4,scy-10),f"{lbl} r={r}",fill=col)

    def _draw_canvas(self,canvas,pil):
        h,w=(self.img_filtered.shape if canvas is self.cv_filt else self.img_orig.shape[:2])
        sc,ox,oy=self._scale(canvas,w,h)
        nw,nh=max(1,int(w*sc)),max(1,int(h*sc))
        tk_img=ImageTk.PhotoImage(pil)
        self._tk_imgs[id(canvas)]=tk_img
        canvas.delete("all")
        canvas.create_image(ox+nw//2,oy+nh//2,image=tk_img)
        canvas.create_line(ox,oy,ox+nw,oy,fill=T["border"])
        canvas.create_line(ox,oy,ox,oy+nh,fill=T["border"])
        step=max(50,int(100/sc))
        for i in range(0,9999,step):
            px=ox+int(i*sc)
            if px>ox+nw: break
            canvas.create_text(px,oy-8,text=str(i),fill=T["fg2"],font=("Helvetica",6))
        for i in range(0,9999,step):
            py=oy+int(i*sc)
            if py>oy+nh: break
            canvas.create_text(ox-18,py,text=str(i),fill=T["fg2"],font=("Helvetica",6))

    # ══════════════════════════════════════════════════
    #  Events
    # ══════════════════════════════════════════════════

    def _img_coord(self,ex,ey):
        if self.img_filtered is None: return (0,0)
        h,w=self.img_filtered.shape; sc,ox,oy=self._scale(self.cv_filt,w,h)
        return (int((ex-ox)/sc),int((ey-oy)/sc))

    def _on_press(self,e):
        if self.img_filtered is None: return
        self._drawing=True; self._pt_start=self._img_coord(e.x,e.y); self._pt_end=self._pt_start
        if self.tool=="inspect": self._show_pixel(self._pt_start)
        elif self.tool=="pan":   self._pan_start=(e.x-self.offset[0],e.y-self.offset[1])

    def _on_drag(self,e):
        if not self._drawing: return
        if self.tool=="pan":
            self.offset=[e.x-self._pan_start[0],e.y-self._pan_start[1]]; self._redraw(); return
        self._pt_end=self._img_coord(e.x,e.y)
        if self.tool in ("measure","roi"): self._redraw()

    def _on_release(self,e):
        if not self._drawing: return
        self._drawing=False; self._pt_end=self._img_coord(e.x,e.y)
        if self.tool=="measure" and self._pt_start:
            p1,p2=self._pt_start,self._pt_end; d=np.linalg.norm(np.array(p1)-np.array(p2))
            self.measure=(p1,p2,d)
            km=f"\nDistance  {d*self.eclipse['km_pp']:,.1f} km" if self.eclipse else ""
            self._log(f"Measurement\n{'-'*24}\nFrom  {p1}\nTo    {p2}\nPixels {d:.2f}{km}")
        elif self.tool=="roi" and self._pt_start:
            x1,y1=self._pt_start; x2,y2=self._pt_end
            x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
            hi,wi=self.img_filtered.shape
            self.roi=(max(0,x1),max(0,y1),min(wi-1,x2),min(hi-1,y2))
            self._calc_roi()
        self._redraw()

    def _on_hover(self,e):
        if self.img_filtered is None: return
        x,y=self._img_coord(e.x,e.y); h,w=self.img_filtered.shape
        v=f"  I={self.img_filtered[y,x]}" if 0<=x<w and 0<=y<h else ""
        self.lbl_cursor.config(text=f"  x={x}  y={y}{v}")

    def _on_zoom(self,e):
        self.zoom=max(0.05,min(self.zoom*(1.12 if e.delta>0 else 0.89),40.0))
        self.lbl_zoom.config(text=f"{self.zoom:.2f}x  "); self._redraw()

    # ══════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════

    def set_tool(self,t):
        self.tool=t; self.measure=None; self._hi_tool(t)
        self.cv_filt.config(cursor={"measure":"crosshair","roi":"sizing","inspect":"tcross","pan":"fleur"}.get(t,"arrow"))
        self._redraw()

    def _hi_tool(self,active):
        for k,b in self._tool_btns.items():
            b.config(bg=T["accent"] if k==active else T["panel2"], fg=T["bg"]    if k==active else T["fg"])

    def _clear_all(self):
        self.outer_circle=self.inner_circle=self.roi=self.measure=self.eclipse=None
        self._cmd_cat=None
        self._clear_profiles(); self._cmd_welcome(); self._redraw()
        self._status("Cleared",T["fg2"])

    def _calc_roi(self):
        x1,y1,x2,y2=self.roi; crop=self.img_filtered[y1:y2,x1:x2]
        if crop.size==0: return
        km=""
        if self.eclipse:
            k=self.eclipse["km_pp"]; km=f"\nW {(x2-x1)*k:,.1f} km  H {(y2-y1)*k:,.1f} km"
        self._log(f"ROI\n{'-'*24}\n({x1},{y1})–({x2},{y2})\n{x2-x1}×{y2-y1} px{km}\n" f"Mean {crop.mean():.1f}  Std {crop.std():.1f}\nRange {crop.min()}–{crop.max()}")

    def _show_pixel(self,pt):
        x,y=pt; h,w=self.img_filtered.shape
        if not(0<=x<w and 0<=y<h): return
        v=self.img_filtered[y,x]
        km=(f"\nDist Moon {math.hypot(x-self.eclipse['cx'],y-self.eclipse['cy'])*self.eclipse['km_pp']:,.1f} km"
            if self.eclipse else "")
        self._log(f"Pixel ({x}, {y})\n{'-'*24}\nI={v}  ({v/255:.3f}){km}")

    def _log(self,msg):
        self.info_box.config(state=tk.NORMAL)
        self.info_box.delete(1.0,tk.END)
        self.info_box.insert(tk.END,msg)
        self.info_box.config(state=tk.DISABLED)

    def _log_circles(self):
        lines=["Circle Detection\n"+"-"*24]
        for c,lbl in [(self.outer_circle,"Outer"),(self.inner_circle,"Inner")]:
            if c: lines.append(f"{lbl}\n  center ({c[0]},{c[1]})\n  r={c[2]}px\n  area {math.pi*c[2]**2:,.0f}px²")
        self._log("\n".join(lines))

    def _log_eclipse(self):
        r=self.eclipse
        self._log(
            f"Eclipse Analysis\n{'='*26}\n"
            f"km/pixel  {r['km_pp']:.4f}\n\n"
            f"OUTER  (Umbra)\n  r  {r['o_r']:.1f}px  {r['o_r_km']:,.1f}km\n  area  {r['o_area']:,.0f}km²\n\n"
            f"INNER  (Moon limb)\n  r  {r['i_r']:.1f}px  {r['i_r_km']:,.1f}km\n\n"
            f"NASA theoretical\n  r  {r['theo_r_km']:,.1f}km  area  {r['theo_area']:,.0f}km²\n\n"
            f"Radius error  {r['err_r']:.3f}%\n"
            f"Area error    {r['err_area']:.3f}%\n"
            f"Shadow cover  {r['dark_pct']:.2f}%\n"
            f"RANSAC  {'on' if HAS_SKIMAGE else 'off'}")

    def _status(self,msg,col=None):
        self.lbl_status.config(text=msg,fg=col or T["fg2"])


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()