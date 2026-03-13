"""
Feature Map & Solar Eclipse Analyzer
ผู้พัฒนา: วิทชภณ พวงแก้ว
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
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.optimize import least_squares

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
EARTH_MASS_KG    = 5.972e24

# ── Dark Theme ──────────────────────────────────────────
T = {
    "bg":       "#0e0f14",
    "panel":    "#161820",
    "panel2":   "#1c1e28",
    "border":   "#2a2d3e",
    "fg":       "#d4d8f0",
    "fg2":      "#7880a0",
    "accent":   "#4f8fff",
    "accent2":  "#a78bfa",
    "success":  "#34d399",
    "warning":  "#fbbf24",
    "danger":   "#f87171",
    "canvas":   "#060709",
    "plot_bg":  "#0e0f14",
    "grid":     "#1e2030",
    "outer":    "#4f8fff",
    "inner":    "#f472b6",
    "button":   "#1c1e28",
    "button_fg": "#d4d8f0",
}

FILTERS = ["Cross-Correlation", "Sobel Edge", "Laplacian", "Sharpen"]


# ══════════════════════════════════════════════════════
#  Core algorithms
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
        r_pen  = (c[2]-expected_r)*0.1 if expected_r else 0.0
        c_pen  = 0.0
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
        c = cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,100,param1=100,param2=50,
                             minRadius=50,maxRadius=int(min(img.shape[:2])/2))
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
    """ตรวจจับขอบวงนอก (Umbra) และวงใน (Moon limb) แยกกัน"""
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

        # INNER: ขอบดวงจันทร์ (บริเวณ r_moon)
        inner_zone = (rv >= r_moon*0.6) & (rv <= r_moon*1.1)
        iz = np.where(inner_zone)[0]
        if len(iz) >= 5:
            ig = sg[iz]
            ii = np.argmin(ig)
            if ig[ii] < -0.3:
                bi = iz[ii]
                sw,ew = max(0,bi-5), min(len(gd),bi+6)
                if ew-sw >= 3:
                    xidx = np.arange(sw,ew)
                    fi   = interp1d(xidx,gd[sw:ew],kind='cubic',fill_value='extrapolate')
                    xn   = np.linspace(sw,ew-1,(ew-sw)*100)
                    rfr  = rv[0]+xn[np.argmin(fi(xn))]
                    inner_pts.append([cx+rfr*np.cos(rad), cy+rfr*np.sin(rad)])
                    inner_profiles.append(sm.copy())
                    if ig[ii] < min_inner:
                        min_inner = ig[ii]
                        best_inner_ray = (sm.copy(), gd.copy(), rv.copy(), rfr)

        # OUTER: ขอบเงา Umbra
        mask = np.ones_like(sg, dtype=bool)
        es,ee = int(0.95*r_moon), int(1.05*r_moon)
        if ee<len(mask): mask[es:ee]=False
        mask[:int(0.15*r_moon)]=False
        vi = np.where(mask)[0]
        if len(vi) < 10: continue
        mg = sg[vi]
        mi = np.argmin(mg)
        if mg[mi] > -0.5: continue
        bri = vi[mi]
        sw,ew = max(0,bri-5), min(len(gd),bri+6)
        if ew-sw < 3: continue
        xidx = np.arange(sw,ew)
        fo   = interp1d(xidx,gd[sw:ew],kind='cubic',fill_value='extrapolate')
        xn   = np.linspace(sw,ew-1,(ew-sw)*100)
        rfr  = rv[0]+xn[np.argmin(fo(xn))]
        outer_pts.append([cx+rfr*np.cos(rad), cy+rfr*np.sin(rad)])
        outer_profiles.append(sm.copy())
        if mg[mi] < min_outer:
            min_outer = mg[mi]
            best_outer_ray = (sm.copy(), gd.copy(), rv.copy(), rfr)

    outer_pts = np.array(outer_pts) if outer_pts else np.empty((0,2))
    inner_pts = np.array(inner_pts) if inner_pts else np.empty((0,2))

    theo_r_km  = R_e - (R_s-R_e)*(D_m/D_s)
    theo_area  = math.pi*theo_r_km**2
    exp_r_px   = theo_r_km/km_pp

    o_cx=o_cy=o_r = 0.0
    out_inlier = outer_pts
    out_outlier = np.empty((0,2))
    if len(outer_pts) >= 3:
        if HAS_SKIMAGE:
            mr, inl = ransac(outer_pts,CircleModel,3,3.0,max_trials=1000)
            if mr:
                out_inlier  = outer_pts[inl] if inl.sum()>10 else outer_pts
                out_outlier = outer_pts[~inl]
        o_cx,o_cy,o_r = fit_circle_constrained(out_inlier,exp_r_px,cx,cy,r_moon)

    i_cx=i_cy=i_r = 0.0
    inn_inlier = inner_pts
    if len(inner_pts) >= 3:
        if HAS_SKIMAGE:
            mr2, inl2 = ransac(inner_pts,CircleModel,3,2.0,max_trials=500)
            if mr2: inn_inlier = inner_pts[inl2] if inl2.sum()>10 else inner_pts
        i_cx,i_cy,i_r = fit_circle_taubin(inn_inlier[:,0], inn_inlier[:,1])

    o_r_km   = o_r*km_pp if o_r else 0
    o_area   = math.pi*o_r_km**2 if o_r_km else 0
    i_r_km   = i_r*km_pp if i_r else 0
    err_r    = abs(o_r_km-theo_r_km)/theo_r_km*100 if o_r_km else 0
    err_area = abs(o_area-theo_area)/theo_area*100  if o_area else 0

    moon_mask = np.zeros(gray.shape[:2],np.uint8)
    cv2.circle(moon_mask,(int(cx),int(cy)),int(r_moon*0.95),255,-1)
    mp = cv2.GaussianBlur(gray,(9,9),0)[moon_mask==255]
    dark_pct = float(np.sum(mp<np.percentile(mp,95)*0.35)/len(mp)*100) if len(mp) else 0.0

    pfl = min((len(p) for p in outer_profiles), default=0) if outer_profiles else 0
    outer_mean = np.mean([p[:pfl] for p in outer_profiles],axis=0) if pfl>10 else np.array([])
    pfl2 = min((len(p) for p in inner_profiles), default=0) if inner_profiles else 0
    inner_mean = np.mean([p[:pfl2] for p in inner_profiles],axis=0) if pfl2>10 else np.array([])

    return {
        "cx":cx,"cy":cy,"r_moon":r_moon,"km_pp":km_pp,
        "o_cx":o_cx,"o_cy":o_cy,"o_r":o_r,"o_r_km":o_r_km,"o_area":o_area,
        "out_inlier":out_inlier,"out_outlier":out_outlier,
        "best_outer_ray":best_outer_ray,"outer_mean":outer_mean,
        "i_cx":i_cx,"i_cy":i_cy,"i_r":i_r,"i_r_km":i_r_km,
        "inn_inlier":inn_inlier,"best_inner_ray":best_inner_ray,"inner_mean":inner_mean,
        "theo_r_km":theo_r_km,"theo_area":theo_area,
        "err_r":err_r,"err_area":err_area,"dark_pct":dark_pct,
        "img_bgr":img,
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
#  Main App
# ══════════════════════════════════════════════════════

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Feature Map & Eclipse Analyzer  —  วิทชภณ พวงแก้ว")
        self.root.geometry("1720x960")
        self.root.configure(bg=T["bg"])

        self.img_orig      = None
        self.img_filtered  = None
        self.outer_circle  = None
        self.inner_circle  = None
        self.roi           = None
        self.measure       = None
        self.eclipse       = None
        self.eclipse_paths = []

        self.zoom   = 1.0
        self.offset = [0, 0]
        self.tool   = "measure"

        self._drawing   = False
        self._pt_start  = None
        self._pt_end    = None
        self._pan_start = (0,0)
        self._pending   = False
        self._tk_imgs   = {}

        self._build()

    # ──────────────────────────────────────────────────
    #  UI
    # ──────────────────────────────────────────────────

    def _build(self):
        self._build_header()
        self._build_body()
        self._style()

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=T["panel"], height=52)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        def sep():
            tk.Frame(hdr, bg=T["border"], width=1).pack(
                side=tk.LEFT, fill=tk.Y, padx=6, pady=10)

        self._hbtn(hdr, "Open Image",     self.load_image,     "accent" ).pack(side=tk.LEFT, padx=(10,0), pady=10)
        self._hbtn(hdr, "Open Eclipse",   self.eclipse_open,   "accent" ).pack(side=tk.LEFT, padx=4,      pady=10)
        sep()

        tk.Label(hdr, text="Filter", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(0,3))
        self.filter_var = tk.StringVar(value=FILTERS[0])
        cb = ttk.Combobox(hdr, textvariable=self.filter_var,
                          values=FILTERS, width=15, state="readonly",
                          font=("Helvetica", 9))
        cb.pack(side=tk.LEFT, pady=12)
        cb.bind("<<ComboboxSelected>>", lambda _: self.refresh_filter())
        sep()

        self._tool_btns = {}
        for key, label in [("measure","Measure"),("roi","ROI"),
                            ("inspect","Inspect"),("pan","Pan")]:
            b = self._hbtn(hdr, label, lambda k=key: self.set_tool(k), "tool")
            b.pack(side=tk.LEFT, padx=2, pady=10)
            self._tool_btns[key] = b
        sep()

        self._hbtn(hdr, "Detect Circles",  self.detect_circles,  "success").pack(side=tk.LEFT, padx=4,  pady=10)
        self._hbtn(hdr, "Sobel Fit",        self.sobel_fit,        "tool"   ).pack(side=tk.LEFT, padx=2,  pady=10)
        sep()

        self._hbtn(hdr, "Analyze Eclipse",  self.eclipse_analyze,  "warning").pack(side=tk.LEFT, padx=4,  pady=10)
        tk.Label(hdr, text="Limb u", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 8)).pack(side=tk.LEFT, padx=(8,2))
        self.limb_u = tk.Scale(hdr, from_=0, to=100, orient=tk.HORIZONTAL,
                               length=70, bg=T["panel"], fg=T["fg"],
                               highlightthickness=0, troughcolor=T["border"],
                               font=("Helvetica", 7), bd=0, sliderlength=12)
        self.limb_u.set(80)
        self.limb_u.pack(side=tk.LEFT, pady=10)
        sep()

        self.lbl_status = tk.Label(hdr, text="No image loaded",
                                   bg=T["panel"], fg=T["fg2"],
                                   font=("Helvetica", 8))
        self.lbl_status.pack(side=tk.LEFT, padx=8)

        self._hbtn(hdr, "Clear", self._clear_all, "danger").pack(
            side=tk.RIGHT, padx=10, pady=10)

    def _build_body(self):
        body = tk.Frame(self.root, bg=T["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(3,0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=3)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        lf_left = self._lframe(body, "Original")
        lf_left.grid(row=0, column=0, sticky="nsew", padx=(0,3))
        self.cv_orig = tk.Canvas(lf_left, bg=T["canvas"], highlightthickness=0)
        self.cv_orig.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        lf_mid = self._lframe(body, "Filtered")
        lf_mid.grid(row=0, column=1, sticky="nsew", padx=3)
        self.cv_filt = tk.Canvas(lf_mid, bg=T["canvas"], highlightthickness=0)
        self.cv_filt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._lbl_filter = lf_mid.winfo_children()[0]

        right = tk.Frame(body, bg=T["bg"])
        right.grid(row=0, column=2, sticky="nsew", padx=(3,0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        lf_info = self._lframe(right, "Analysis")
        lf_info.grid(row=0, column=0, sticky="nsew", pady=(0,3))
        self.info_box = tk.Text(lf_info, font=("Menlo", 8),
                                state=tk.DISABLED, wrap=tk.WORD,
                                bg=T["panel2"], fg=T["fg"],
                                relief=tk.FLAT, bd=0, padx=8, pady=6)
        self.info_box.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        lf_graph = self._lframe(right, "Profiles")
        lf_graph.grid(row=1, column=0, sticky="nsew")
        self._build_graph_panel(lf_graph)

        self.cv_filt.bind("<ButtonPress-1>",   self._on_press)
        self.cv_filt.bind("<B1-Motion>",        self._on_drag)
        self.cv_filt.bind("<ButtonRelease-1>", self._on_release)
        self.cv_filt.bind("<Motion>",           self._on_hover)
        for cv in (self.cv_filt, self.cv_orig):
            cv.bind("<MouseWheel>", self._on_zoom)

        sb = tk.Frame(self.root, bg=T["button"], height=20)
        sb.pack(fill=tk.X)
        sb.pack_propagate(False)
        self.lbl_cursor = tk.Label(sb, text="  x=—  y=—", anchor="w",bg=T["button"], fg=T["button_fg"], font=("Menlo", 7))
        self.lbl_cursor.pack(side=tk.LEFT)
        self.lbl_zoom = tk.Label(sb, text="1.00x  ", anchor="e",bg=T["button"], fg=T["button_fg"], font=("Menlo", 7))
        self.lbl_zoom.pack(side=tk.RIGHT)

    def _build_graph_panel(self, parent):
        self._fig = Figure(figsize=(5, 5.5), facecolor=T["plot_bg"])
        self._fig.subplots_adjust(hspace=0.6, wspace=0.4,top=0.92, bottom=0.08, left=0.13, right=0.97)
        self._ax_outer  = self._fig.add_subplot(2, 2, 1)
        self._ax_inner  = self._fig.add_subplot(2, 2, 2)
        self._ax_radial = self._fig.add_subplot(2, 2, 3)
        self._ax_hist   = self._fig.add_subplot(2, 2, 4)
        for ax, t in [(self._ax_outer,"Outer Edge"),(self._ax_inner,"Inner Edge"),(self._ax_radial,"Radial Profile"),(self._ax_hist,"Histogram")]:dax(ax, t)
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_mpl.draw()
        self._canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ──────────────────────────────────────────────────
    #  Widgets
    # ──────────────────────────────────────────────────

    def _lframe(self, parent, title):
        f = tk.Frame(parent, bg=T["panel"],highlightbackground=T["border"], highlightthickness=1)
        tk.Label(f, text=title.upper(), bg=T["panel"], fg=T["fg2"],font=("TH Sarabun New", 16, "bold"), padx=6, pady=3).pack(anchor="w")
        return f

    def _hbtn(self, parent, text, cmd, style="tool"):
        colors = {
            "accent":  (T["accent"],  T["bg"]),
            "success": (T["success"], T["bg"]),
            "warning": (T["warning"], T["bg"]),
            "danger":  (T["danger"],  T["bg"]),
            "tool":    (T["panel2"],  T["fg"]),
        }
        bg, fg = colors.get(style, (T["panel2"], T["fg"]))
        return tk.Button(parent, text=text, command=cmd,bg=bg, fg=fg, activebackground=T["border"],activeforeground=T["fg"],font=("Helvetica", 9), relief=tk.FLAT,cursor="hand2", padx=10, pady=3,bd=0, highlightthickness=0)

    def _style(self):
        s = ttk.Style(); s.theme_use("clam")
        s.configure("TCombobox", fieldbackground=T["panel2"],
                    background=T["panel2"], foreground=T["fg"],
                    selectbackground=T["panel2"], selectforeground=T["fg"],
                    arrowcolor=T["fg2"])
        self._hi_tool("measure")

    # ──────────────────────────────────────────────────
    #  Load / Filter
    # ──────────────────────────────────────────────────

    def load_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("All","*.*")])
        if not p: return
        raw = cv2.imread(p)
        if raw is None: messagebox.showerror("Error","Cannot open file"); return
        self.img_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self._reset(); self.refresh_filter()

    def eclipse_open(self):
        paths = filedialog.askopenfilenames(
            title="Open eclipse image(s)",
            filetypes=[("Images","*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),("All","*.*")])
        if not paths: return
        self.eclipse_paths = list(paths)
        raw = cv2.imread(paths[0])
        if raw is None: return
        self.img_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self._reset(); self.refresh_filter()
        self._status(f"{len(paths)} frame(s) ready", T["warning"])

    def refresh_filter(self):
        if self.img_orig is None: return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        self.img_filtered = apply_filter(gray, self.filter_var.get())
        h,w = self.img_filtered.shape
        self._lbl_filter.config(text=f"FILTERED  {self.filter_var.get()}  {w}x{h}")
        self._redraw()

    def _reset(self):
        self.outer_circle = self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self.zoom, self.offset = 1.0, [0,0]
        self._clear_plots()
        self._status("Image loaded", T["fg2"])

    # ──────────────────────────────────────────────────
    #  Circle detection
    # ──────────────────────────────────────────────────

    def detect_circles(self):
        if self.img_orig is None:
            messagebox.showwarning("","Open an image first"); return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(9,9),2)
        h,w  = gray.shape
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2,
                                   minDist=max(20,min(h,w)//8),
                                   param1=60, param2=30,
                                   minRadius=max(8,min(h,w)//30),
                                   maxRadius=min(h,w)//2)
        if circles is None:
            messagebox.showinfo("","No circles detected"); return
        circles = np.round(circles[0]).astype(int)
        circles = circles[circles[:,2].argsort()[::-1]]
        self.outer_circle = tuple(circles[0])
        self.inner_circle = tuple(circles[1]) if len(circles)>=2 else None
        self._log_circles()
        self._update_basic_plots()
        self._redraw()
        self._status(
            f"Outer r={circles[0][2]}px" +
            (f"  Inner r={circles[1][2]}px" if len(circles)>=2 else ""),
            T["success"])

    def sobel_fit(self):
        if self.img_orig is None: return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),1.5)
        sx   = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
        sy   = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
        mag  = np.sqrt(sx**2+sy**2)
        thr_o = np.percentile(mag, 97)
        thr_i = np.percentile(mag, 94)
        for thr, hi, attr in [(thr_o,None,"outer_circle"),(thr_i,thr_o,"inner_circle")]:
            if hi is None: mask = (mag>=thr).astype(np.uint8)
            else:          mask = ((mag>=thr)&(mag<hi)).astype(np.uint8)
            ys,xs = np.where(mask>0)
            if len(xs)<6: continue
            try:
                cx,cy,r = fit_circle_taubin(xs,ys)
                setattr(self, attr, (int(round(cx)),int(round(cy)),int(round(r))))
            except: pass
        self._log_circles()
        self._update_basic_plots()
        self._redraw()

    # ──────────────────────────────────────────────────
    #  Eclipse analysis
    # ──────────────────────────────────────────────────

    def eclipse_analyze(self):
        if not self.eclipse_paths and self.img_orig is None:
            messagebox.showwarning("","Open eclipse image first"); return
        self._status("Analyzing...", T["warning"]); self.root.update()
        try:
            u = self.limb_u.get()/100.0
            if len(self.eclipse_paths) > 1:
                img, info = align_and_median_combine(self.eclipse_paths)
                if img is None: raise RuntimeError("Alignment failed")
                cx,cy,r_moon = info
            else:
                img = cv2.cvtColor(self.img_orig,cv2.COLOR_RGB2BGR) \
                      if self.img_orig is not None else cv2.imread(self.eclipse_paths[0])
                if img is None: raise RuntimeError("Cannot load image")
                g = cv2.GaussianBlur(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(5,5),0)
                c = cv2.HoughCircles(g,cv2.HOUGH_GRADIENT,1.2,100,param1=100,param2=50,
                                     minRadius=50,maxRadius=int(min(img.shape[:2])/2))
                if c is None: raise RuntimeError("Moon not detected")
                x = np.uint16(np.around(c))[0,0]
                cx,cy,r_moon = int(x[0]),int(x[1]),int(x[2])

            res = detect_rings(img, cx, cy, r_moon, u)
            if res is None: raise RuntimeError("No umbra edge found")
            self.eclipse = res
            if res["o_r"] > 0:
                self.outer_circle = (int(res["o_cx"]),int(res["o_cy"]),int(res["o_r"]))
            if res["i_r"] > 0:
                self.inner_circle = (int(res["i_cx"]),int(res["i_cy"]),int(res["i_r"]))
            self._log_eclipse()
            self._update_eclipse_plots()
            self._redraw()
            self._status(
                f"Outer {res['o_r_km']:,.0f} km  Inner {res['i_r_km']:,.0f} km  err {res['err_r']:.2f}%",
                T["success"])
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self._status("Analysis failed", T["danger"])

    # ──────────────────────────────────────────────────
    #  Graph updates
    # ──────────────────────────────────────────────────

    def _clear_plots(self):
        for ax,t in [(self._ax_outer,"Outer Edge"),(self._ax_inner,"Inner Edge"),
                     (self._ax_radial,"Radial Profile"),(self._ax_hist,"Histogram")]:
            ax.cla(); dax(ax, t)
        self._canvas_mpl.draw_idle()

    def _radial_profile_for(self, circle, gray):
        """คืน (dr, mean_p, std_p, grad_p) รอบวงกลมที่กำหนด"""
        cx,cy,r = circle
        h,w = gray.shape
        sw = int(r*0.45)
        dr = np.arange(-sw, sw+1, dtype=float)
        profiles = np.full((180, len(dr)), np.nan)
        for i in range(180):
            ca = math.cos(math.pi*i/180)
            sa = math.sin(math.pi*i/180)
            for j,d in enumerate(dr):
                px,py = int(round(cx+(r+d)*ca)), int(round(cy+(r+d)*sa))
                if 0<=px<w and 0<=py<h: profiles[i,j] = gray[py,px]
        mp = np.nanmean(profiles, axis=0)
        sp = np.nanstd(profiles,  axis=0)
        gp = np.gradient(mp)
        return dr, mp, sp, gp

    def _update_basic_plots(self):
        if self.img_orig is None: return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)

        for circle, ax, col, label in [
            (self.outer_circle, self._ax_outer, T["outer"], "Outer"),
            (self.inner_circle, self._ax_inner, T["inner"], "Inner"),
        ]:
            ax.cla(); dax(ax, f"{label} Edge", "dr (px)", "I")
            if circle is None: continue
            dr,mp,sp,gp = self._radial_profile_for(circle, gray)
            ei = int(np.argmin(gp))
            ax.plot(dr, mp, color=col, lw=1.4, label="Mean")
            ax.fill_between(dr, mp-sp, mp+sp, alpha=0.12, color=col)
            ax.plot(dr, gp*3, color=T["fg2"], lw=0.9, ls="--", label="Grad x3")
            ax.axvline(dr[ei], color=T["warning"], lw=1.2, ls=":",
                       label=f"{dr[ei]:+.1f}px")
            ax.legend(fontsize=6, labelcolor=T["fg"], loc="upper right")

        self._ax_radial.cla(); dax(self._ax_radial, "Radial Profile", "r (px)", "I")
        for circle, col, label in [(self.outer_circle,T["outer"],"outer"),
                                   (self.inner_circle,T["inner"],"inner")]:
            if circle is None: continue
            cx,cy,r = circle
            rv = np.arange(0, int(r*1.5))
            profs = []
            for i in range(0,360,3):
                ca,sa = math.cos(math.pi*i/180), math.sin(math.pi*i/180)
                xi = np.clip((cx+rv*ca).astype(int),0,gray.shape[1]-1)
                yi = np.clip((cy+rv*sa).astype(int),0,gray.shape[0]-1)
                profs.append(gray[yi,xi].astype(float))
            mp = np.mean(profs,axis=0)
            self._ax_radial.plot(rv, mp, color=col, lw=1.3, label=label)
            self._ax_radial.axvline(r, color=col, lw=1, ls="--", alpha=0.5)
        self._ax_radial.legend(fontsize=6, labelcolor=T["fg"])

        self._ax_hist.cla(); dax(self._ax_hist, "Histogram", "I", "n")
        if self.img_filtered is not None:
            self._ax_hist.hist(self.img_filtered.flatten(), bins=64,
                               range=(0,256), color=T["accent2"], alpha=0.8, edgecolor="none")

        self._canvas_mpl.draw_idle()

    def _update_eclipse_plots(self):
        r = self.eclipse
        if r is None: return
        gray = cv2.cvtColor(r["img_bgr"], cv2.COLOR_BGR2GRAY)

        # OUTER edge profile
        self._ax_outer.cla(); dax(self._ax_outer, "Outer Edge (Umbra)", "dr (px)", "I")
        if self.outer_circle:
            dr,mp,sp,gp = self._radial_profile_for(self.outer_circle, gray)
            ei = int(np.argmin(gp))
            self._ax_outer.plot(dr, mp, color=T["outer"], lw=1.4)
            self._ax_outer.fill_between(dr, mp-sp, mp+sp, alpha=0.12, color=T["outer"])
            self._ax_outer.plot(dr, gp*3, color=T["fg2"], lw=0.9, ls="--")
            self._ax_outer.axvline(dr[ei], color=T["warning"], lw=1.3, ls=":",
                                   label=f"edge {dr[ei]:+.1f}px")
            self._ax_outer.legend(fontsize=6, labelcolor=T["fg"])

        # INNER edge profile
        self._ax_inner.cla(); dax(self._ax_inner, "Inner Edge (Moon limb)", "dr (px)", "I")
        if self.inner_circle:
            dr2,mp2,sp2,gp2 = self._radial_profile_for(self.inner_circle, gray)
            ei2 = int(np.argmin(gp2))
            self._ax_inner.plot(dr2, mp2, color=T["inner"], lw=1.4)
            self._ax_inner.fill_between(dr2, mp2-sp2, mp2+sp2, alpha=0.12, color=T["inner"])
            self._ax_inner.plot(dr2, gp2*3, color=T["fg2"], lw=0.9, ls="--")
            self._ax_inner.axvline(dr2[ei2], color=T["warning"], lw=1.3, ls=":",
                                   label=f"edge {dr2[ei2]:+.1f}px")
            self._ax_inner.legend(fontsize=6, labelcolor=T["fg"])

        # Full radial profile
        self._ax_radial.cla(); dax(self._ax_radial, "Full Radial", "r (px)", "I")
        cx,cy,rm = r["cx"],r["cy"],r["r_moon"]
        rv2 = np.arange(0, int(1.8*rm))
        profs = []
        for i in range(0,360,3):
            ca,sa = math.cos(math.pi*i/180), math.sin(math.pi*i/180)
            xi = np.clip((cx+rv2*ca).astype(int),0,gray.shape[1]-1)
            yi = np.clip((cy+rv2*sa).astype(int),0,gray.shape[0]-1)
            profs.append(gray[yi,xi].astype(float))
        mp3 = np.mean(profs,axis=0)
        sp3 = np.std(profs,axis=0)
        self._ax_radial.plot(rv2, mp3, color=T["accent"], lw=1.4)
        self._ax_radial.fill_between(rv2, mp3-sp3, mp3+sp3, alpha=0.1, color=T["accent"])
        self._ax_radial.axvline(rm, color=T["inner"], lw=1.1, ls="--", label=f"Moon r={rm}px")
        if r["o_r"]>0:
            self._ax_radial.axvline(r["o_r"], color=T["outer"], lw=1.1, ls="--",
                                    label=f"Umbra r={r['o_r']:.0f}px")
        self._ax_radial.legend(fontsize=6, labelcolor=T["fg"])

        # Histogram
        self._ax_hist.cla(); dax(self._ax_hist, "Histogram", "I", "n")
        gf = gray.flatten()
        self._ax_hist.hist(gf, bins=64, range=(0,256),
                           color=T["accent2"], alpha=0.8, edgecolor="none")
        thresh = np.percentile(gf,95)*0.35
        self._ax_hist.axvline(thresh, color=T["danger"], lw=1, ls="--", label="shadow")
        self._ax_hist.legend(fontsize=6, labelcolor=T["fg"])

        self._canvas_mpl.draw_idle()

    # ──────────────────────────────────────────────────
    #  Canvas render
    # ──────────────────────────────────────────────────

    def _redraw(self):
        if not self._pending:
            self._pending = True
            self.root.after(16, self._do_redraw)

    def _do_redraw(self):
        self._pending = False
        if self.img_filtered is None: return
        self._draw_canvas(self.cv_orig, self._make_orig())
        self._draw_canvas(self.cv_filt, self._make_filt())

    def _scale(self, canvas, iw, ih):
        cw = canvas.winfo_width()  or 500
        ch = canvas.winfo_height() or 400
        sc = min(cw/iw, ch/ih) * 0.92 * self.zoom
        ox = cw//2 + self.offset[0] - int(iw*sc)//2
        oy = ch//2 + self.offset[1] - int(ih*sc)//2
        return sc, ox, oy

    def _make_orig(self):
        h,w = self.img_orig.shape[:2]
        sc,ox,oy = self._scale(self.cv_orig, w, h)
        nw,nh = max(1,int(w*sc)), max(1,int(h*sc))
        pil  = Image.fromarray(self.img_orig).resize((nw,nh), Image.BILINEAR)
        draw = ImageDraw.Draw(pil)
        self._draw_rings(draw, sc)
        return pil

    def _make_filt(self):
        h,w = self.img_filtered.shape
        sc,ox,oy = self._scale(self.cv_filt, w, h)
        nw,nh = max(1,int(w*sc)), max(1,int(h*sc))
        pil  = Image.fromarray(self.img_filtered).resize((nw,nh), Image.BILINEAR)
        draw = ImageDraw.Draw(pil)
        if self.measure:
            p1,p2,d = self.measure
            s1=(int(p1[0]*sc),int(p1[1]*sc)); s2=(int(p2[0]*sc),int(p2[1]*sc))
            draw.line([s1,s2], fill="#ffffff", width=2)
            for pt in (s1,s2): draw.ellipse([pt[0]-4,pt[1]-4,pt[0]+4,pt[1]+4],fill=T["danger"])
            km = f"  {d*self.eclipse['km_pp']:.0f}km" if self.eclipse else ""
            draw.text(((s1[0]+s2[0])//2+4,(s1[1]+s2[1])//2-12),
                      f"{d:.1f}px{km}", fill=T["warning"])
        if self.roi:
            x1,y1,x2,y2 = self.roi
            draw.rectangle([int(x1*sc),int(y1*sc),int(x2*sc),int(y2*sc)],
                           outline=T["success"], width=2)
        self._draw_rings(draw, sc)
        if self._drawing and self._pt_start and self._pt_end:
            s1=(int(self._pt_start[0]*sc),int(self._pt_start[1]*sc))
            s2=(int(self._pt_end[0]*sc),  int(self._pt_end[1]*sc))
            if self.tool=="roi":       draw.rectangle([s1,s2],outline=T["warning"],width=1)
            elif self.tool=="measure": draw.line([s1,s2],fill=T["accent"],width=1)
        return pil

    def _draw_rings(self, draw, sc):
        for circle, col, label in [
            (self.outer_circle, T["outer"], "outer"),
            (self.inner_circle, T["inner"], "inner"),
        ]:
            if circle is None: continue
            cx,cy,r = circle
            scx,scy,sr = int(cx*sc),int(cy*sc),int(r*sc)
            pts = []
            for i in range(361):
                a = 2*math.pi*i/360
                pts.append((scx+int(sr*math.cos(a)), scy+int(sr*math.sin(a))))
            draw.line(pts, fill=col, width=2)
            draw.line([(scx-8,scy),(scx+8,scy)], fill=col, width=1)
            draw.line([(scx,scy-8),(scx,scy+8)], fill=col, width=1)
            draw.text((scx+sr+4, scy-10), f"{label} r={r}", fill=col)

    def _draw_canvas(self, canvas, pil):
        h,w = (self.img_filtered.shape if canvas is self.cv_filt
               else self.img_orig.shape[:2])
        sc,ox,oy = self._scale(canvas, w, h)
        nw,nh = max(1,int(w*sc)), max(1,int(h*sc))
        tk_img = ImageTk.PhotoImage(pil)
        self._tk_imgs[id(canvas)] = tk_img
        canvas.delete("all")
        canvas.create_image(ox+nw//2, oy+nh//2, image=tk_img)
        canvas.create_line(ox,oy,ox+nw,oy, fill=T["border"])
        canvas.create_line(ox,oy,ox,oy+nh, fill=T["border"])
        step = max(50, int(100/sc))
        for i in range(0,9999,step):
            px=ox+int(i*sc)
            if px>ox+nw: break
            canvas.create_text(px,oy-8,text=str(i),fill=T["fg2"],font=("Helvetica",6))
        for i in range(0,9999,step):
            py=oy+int(i*sc)
            if py>oy+nh: break
            canvas.create_text(ox-18,py,text=str(i),fill=T["fg2"],font=("Helvetica",6))

    # ──────────────────────────────────────────────────
    #  Events
    # ──────────────────────────────────────────────────

    def _img_coord(self, ex, ey):
        if self.img_filtered is None: return (0,0)
        h,w = self.img_filtered.shape
        sc,ox,oy = self._scale(self.cv_filt, w, h)
        return (int((ex-ox)/sc), int((ey-oy)/sc))

    def _on_press(self, e):
        if self.img_filtered is None: return
        self._drawing  = True
        self._pt_start = self._img_coord(e.x,e.y)
        self._pt_end   = self._pt_start
        if self.tool=="inspect": self._show_pixel(self._pt_start)
        elif self.tool=="pan":   self._pan_start=(e.x-self.offset[0],e.y-self.offset[1])

    def _on_drag(self, e):
        if not self._drawing: return
        if self.tool=="pan":
            self.offset=[e.x-self._pan_start[0],e.y-self._pan_start[1]]
            self._redraw(); return
        self._pt_end=self._img_coord(e.x,e.y)
        if self.tool in ("measure","roi"): self._redraw()

    def _on_release(self, e):
        if not self._drawing: return
        self._drawing=False; self._pt_end=self._img_coord(e.x,e.y)
        if self.tool=="measure" and self._pt_start:
            p1,p2=self._pt_start,self._pt_end
            d=np.linalg.norm(np.array(p1)-np.array(p2))
            self.measure=(p1,p2,d)
            km = f"\nDistance  {d*self.eclipse['km_pp']:,.1f} km" if self.eclipse else ""
            self._log(f"Measurement\n{'-'*24}\nFrom  {p1}\nTo    {p2}\nPixels {d:.2f}{km}")
        elif self.tool=="roi" and self._pt_start:
            x1,y1=self._pt_start; x2,y2=self._pt_end
            x1,x2=sorted([x1,x2]); y1,y2=sorted([y1,y2])
            hi,wi=self.img_filtered.shape
            self.roi=(max(0,x1),max(0,y1),min(wi-1,x2),min(hi-1,y2))
            self._calc_roi()
        self._redraw()

    def _on_hover(self, e):
        if self.img_filtered is None: return
        x,y=self._img_coord(e.x,e.y)
        h,w=self.img_filtered.shape
        v = f"  I={self.img_filtered[y,x]}" if 0<=x<w and 0<=y<h else ""
        self.lbl_cursor.config(text=f"  x={x}  y={y}{v}")

    def _on_zoom(self, e):
        self.zoom=max(0.05,min(self.zoom*(1.12 if e.delta>0 else 0.89),40.0))
        self.lbl_zoom.config(text=f"{self.zoom:.2f}x  ")
        self._redraw()

    # ──────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────

    def set_tool(self, t):
        self.tool=t; self.measure=None
        self._hi_tool(t)
        cursors={"measure":"crosshair","roi":"sizing","inspect":"tcross","pan":"fleur"}
        self.cv_filt.config(cursor=cursors.get(t,"arrow"))
        self._redraw()

    def _hi_tool(self, active):
        for k,b in self._tool_btns.items():
            b.config(bg=T["accent"] if k==active else T["panel2"],
                     fg=T["bg"]    if k==active else T["fg"])

    def _clear_all(self):
        self.outer_circle=self.inner_circle=self.roi=self.measure=self.eclipse=None
        self._clear_plots(); self._redraw()
        self._status("Cleared", T["fg2"])

    def _calc_roi(self):
        x1,y1,x2,y2=self.roi
        crop=self.img_filtered[y1:y2,x1:x2]
        if crop.size==0: return
        km = ""
        if self.eclipse:
            k=self.eclipse["km_pp"]
            km=f"\nW {(x2-x1)*k:,.1f} km  H {(y2-y1)*k:,.1f} km"
        self._log(f"ROI\n{'-'*24}\n({x1},{y1}) - ({x2},{y2})\n"
                  f"{x2-x1}x{y2-y1} px{km}\n"
                  f"Mean {crop.mean():.1f}  Std {crop.std():.1f}\n"
                  f"Range {crop.min()}-{crop.max()}")

    def _show_pixel(self, pt):
        x,y=pt; h,w=self.img_filtered.shape
        if not(0<=x<w and 0<=y<h): return
        v=self.img_filtered[y,x]
        km = f"\nDist Moon {math.hypot(x-self.eclipse['cx'],y-self.eclipse['cy'])*self.eclipse['km_pp']:,.1f} km" \
             if self.eclipse else ""
        self._log(f"Pixel ({x}, {y})\n{'-'*24}\nI={v}  ({v/255:.3f}){km}")

    def _log(self, msg):
        self.info_box.config(state=tk.NORMAL)
        self.info_box.delete(1.0,tk.END)
        self.info_box.insert(tk.END, msg)
        self.info_box.config(state=tk.DISABLED)

    def _log_circles(self):
        lines=["Circle Detection\n"+"-"*24]
        for c,label in [(self.outer_circle,"Outer"),(self.inner_circle,"Inner")]:
            if c: lines.append(f"{label}\n  center ({c[0]},{c[1]})\n  r={c[2]}px\n  area {math.pi*c[2]**2:,.0f}px2")
        self._log("\n".join(lines))

    def _log_eclipse(self):
        r=self.eclipse
        self._log(
            f"Eclipse Analysis\n{'='*26}\n"
            f"km/pixel  {r['km_pp']:.4f}\n\n"
            f"OUTER  (Umbra)\n"
            f"  r     {r['o_r']:.1f} px\n"
            f"        {r['o_r_km']:,.1f} km\n"
            f"  area  {r['o_area']:,.0f} km2\n\n"
            f"INNER  (Moon limb)\n"
            f"  r     {r['i_r']:.1f} px\n"
            f"        {r['i_r_km']:,.1f} km\n\n"
            f"NASA theoretical\n"
            f"  r     {r['theo_r_km']:,.1f} km\n"
            f"  area  {r['theo_area']:,.0f} km2\n\n"
            f"Radius error  {r['err_r']:.3f}%\n"
            f"Area error    {r['err_area']:.3f}%\n"
            f"Shadow cover  {r['dark_pct']:.2f}%\n"
            f"RANSAC  {'on' if HAS_SKIMAGE else 'off'}"
        )

    def _status(self, msg, col=None):
        self.lbl_status.config(text=msg, fg=col or T["fg2"])


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()