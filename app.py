"""
app.py — Main UI
Feature Map & Solar Eclipse Analyzer  +  CMD Plot
ครูที่ปรึกษา: วิทชภณ พวงแก้ว
Developed by: Sitthichokthq (https://github.com/Sitthichokthq) and (sitthichokthq.web.app)

แก้ไข:
  - แยก logic การประมวลผลออกมา analysis.py
  - ตัวหนังสือในกราฟอ่านได้ชัดขึ้น (ขนาด + ความสว่าง)
  - ใช้ threading เพื่อให้ UI ไม่ค้าง
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import math
import threading
import queue
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.gridspec as mgridspec
from scipy.ndimage import gaussian_filter1d

# ── Import computation backend ──────────────────────────────────────────────
from analysis import (
    apply_filter, FILTERS,
    fit_circle_taubin, fit_circle_constrained,
    detect_rings, align_and_median_combine,
    phot_background, phot_detect,
    HAS_SKIMAGE,
)

# ── Dark Theme ───────────────────────────────────────────────────────────────
T = {
    "bg":        "#0e0f14",
    "panel":     "#161820",
    "panel2":    "#1c1e28",
    "border":    "#2a2d3e",
    "fg":        "#d4d8f0",   # primary text — bright
    "fg2":       "#9ba3c0",   # secondary text — raised from #7880a0
    "accent":    "#4f8fff",
    "accent2":   "#a78bfa",
    "success":   "#34d399",
    "warning":   "#fbbf24",
    "danger":    "#f87171",
    "canvas":    "#060709",
    "plot_bg":   "#12131a",   # slightly lighter so axes text is readable
    "grid":      "#252840",
    "outer":     "#60aaff",   # brighter outer ring colour
    "inner":     "#f472b6",
    "button":    "#1c1e28",
    "button_fg": "#d4d8f0",
    "tab_sel":   "#252838",
    # Graph-specific overrides — always bright
    "ax_title":  "#e8ecff",   # axis title
    "ax_label":  "#b8bede",   # x/y labels
    "ax_tick":   "#9ba3c0",   # tick labels
    "ax_grid":   "#252840",   # grid lines
}


# ── Axes Styling Helper ───────────────────────────────────────────────────────

def dax(ax, title: str, xl: str = "", yl: str = "", fontsize_title: int = 9):
    """Apply consistent dark-theme styling to a matplotlib Axes object."""
    ax.set_facecolor(T["plot_bg"])
    ax.set_title(title, color=T["ax_title"],
                 fontsize=fontsize_title, fontweight="bold", pad=5)
    ax.set_xlabel(xl, color=T["ax_label"], fontsize=8, labelpad=3)
    ax.set_ylabel(yl, color=T["ax_label"], fontsize=8, labelpad=3)
    ax.tick_params(colors=T["ax_tick"], labelsize=7.5, length=3, width=0.8)
    for sp in ax.spines.values():
        sp.set_color(T["border"])
        sp.set_linewidth(0.8)
    ax.grid(color=T["ax_grid"], lw=0.6, ls="--", alpha=0.8)


def dax_pub(ax, title: str, xl: str = "", yl: str = ""):
    """Light-mode axes for publication export."""
    ax.set_facecolor("white")
    ax.set_title(title, color="black", fontsize=9, fontweight="bold", pad=5)
    ax.set_xlabel(xl, color="#333333", fontsize=8, labelpad=3)
    ax.set_ylabel(yl, color="#333333", fontsize=8, labelpad=3)
    ax.tick_params(colors="black", labelsize=7.5, length=3, width=0.8)
    for sp in ax.spines.values():
        sp.set_color("black")
        sp.set_linewidth(0.8)
    ax.grid(color="#cccccc", lw=0.5, ls="--", alpha=0.6)


def styled_legend(ax, fontsize: int = 8, loc: str = "best"):
    """
    Create a legend with proper dark-theme styling so text is always visible.
    Legend frame: dark panel background, bright border, white text.
    Returns the Legend object.
    """
    leg = ax.legend(
        fontsize=fontsize,
        loc=loc,
        framealpha=0.88,
        edgecolor=T["border"],
        facecolor=T["panel2"],
    )
    # Make legend text bright white so it stands out against the dark frame
    for txt in leg.get_texts():
        txt.set_color("#ffffff")
        txt.set_fontsize(fontsize)
    # Make legend line/patch handles a touch brighter too
    for handle in leg.legend_handles:
        try:
            handle.set_alpha(1.0)
        except Exception:
            pass
    return leg


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════

class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Feature Map & Eclipse Analyzer")
        self.root.geometry("1920x1080")
        self.root.configure(bg=T["bg"])

        self.img_orig     = None
        self.img_filtered = None
        self.outer_circle = None
        self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self.eclipse_paths: list = []

        self.zoom   = 1.0
        self.offset = [0, 0]
        self.tool   = "measure"

        self._drawing   = False
        self._pt_start  = self._pt_end = None
        self._pan_start = (0, 0)
        self._pending   = False
        self._tk_imgs   = {}

        # Worker thread queue for heavy tasks
        self._work_queue: queue.Queue = queue.Queue()
        self._worker_thread = threading.Thread(target=self._worker_loop,
                                               daemon=True)
        self._worker_thread.start()

        # CMD state
        self._cmd_cat    = None
        self._cmd_thresh = tk.DoubleVar(value=4.0)
        self._cmd_ap     = tk.IntVar(value=5)
        self._cmd_tip    = tk.DoubleVar(value=0.0)
        self._cmd_style  = tk.StringVar(value="Dark")

        self._build()

    # ── Background worker thread ─────────────────────────────────────────────

    def _worker_loop(self):
        """Continuously pull tasks from queue and execute them."""
        while True:
            fn, args, callback = self._work_queue.get()
            try:
                result = fn(*args)
                if callback:
                    self.root.after(0, callback, result, None)
            except Exception as exc:
                if callback:
                    self.root.after(0, callback, None, exc)
            finally:
                self._work_queue.task_done()

    def _run_async(self, fn, args=(), callback=None):
        """Submit a task to the background worker thread."""
        self._work_queue.put((fn, args, callback))

    # ══════════════════════════════════════════════════════════════════════════
    #  Build UI
    # ══════════════════════════════════════════════════════════════════════════

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

        self._hbtn(hdr, "Open Image",    self.load_image,    "accent" ).pack(side=tk.LEFT, padx=(6, 0), pady=10)
        sep()

        tk.Label(hdr, text="Filter", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 3))
        self.filter_var = tk.StringVar(value=FILTERS[0])
        cb = ttk.Combobox(hdr, textvariable=self.filter_var,
                          values=FILTERS, width=14, state="readonly",
                          font=("Helvetica", 9))
        cb.pack(side=tk.LEFT, pady=12)
        cb.bind("<<ComboboxSelected>>", lambda _: self.refresh_filter())
        sep()

        self._tool_btns = {}
        for key, lbl in [("measure", "Measure"), ("roi", "ROI"),
                          ("inspect", "Inspect"), ("pan", "Pan")]:
            b = self._hbtn(hdr, lbl, lambda k=key: self.set_tool(k), "tool")
            b.pack(side=tk.LEFT, padx=2, pady=10)
            self._tool_btns[key] = b
        sep()

        self._hbtn(hdr, "Detect Circles", self.detect_circles, "success").pack(side=tk.LEFT, padx=3, pady=10)
        self._hbtn(hdr, "Sobel Fit",       self.sobel_fit,      "tool"  ).pack(side=tk.LEFT, padx=2, pady=10)
        sep()

        self._hbtn(hdr, "Analyze Eclipse", self.eclipse_analyze, "warning").pack(side=tk.LEFT, padx=3, pady=10)
        tk.Label(hdr, text="u", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(6, 1))
        self.limb_u = tk.Scale(hdr, from_=0, to=100, orient=tk.HORIZONTAL,
                               length=60, bg=T["panel"], fg=T["fg"],
                               highlightthickness=0, troughcolor=T["border"],
                               font=("Helvetica", 7), bd=0, sliderlength=12)
        self.limb_u.set(80)
        self.limb_u.pack(side=tk.LEFT, pady=10)
        sep()

        tk.Label(hdr, text="σ:", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(0, 1))
        tk.Spinbox(hdr, from_=1, to=20, increment=0.5, format="%.1f",
                   textvariable=self._cmd_thresh, width=4,
                   font=("Menlo", 8), bg=T["panel2"], fg=T["fg"],
                   buttonbackground=T["panel2"], relief=tk.FLAT, bd=0,
                   highlightthickness=1, highlightbackground=T["border"]
                   ).pack(side=tk.LEFT, pady=12)
        tk.Label(hdr, text="ap:", bg=T["panel"], fg=T["fg2"],
                 font=("Helvetica", 9)).pack(side=tk.LEFT, padx=(4, 1))
        tk.Spinbox(hdr, from_=2, to=20, textvariable=self._cmd_ap, width=3,
                   font=("Menlo", 8), bg=T["panel2"], fg=T["fg"],
                   buttonbackground=T["panel2"], relief=tk.FLAT, bd=0,
                   highlightthickness=1, highlightbackground=T["border"]
                   ).pack(side=tk.LEFT, pady=12)
        self._hbtn(hdr, "▶ Detect Stars", self.run_cmd, "accent2").pack(side=tk.LEFT, padx=4, pady=10)
        sep()

        self.lbl_status = tk.Label(hdr, text="No image loaded",
                                   bg=T["panel"], fg=T["fg2"],
                                   font=("Helvetica", 8))
        self.lbl_status.pack(side=tk.LEFT, padx=6)
        self._hbtn(hdr, "Clear", self._clear_all, "danger").pack(side=tk.RIGHT, padx=8, pady=10)

    def _build_body(self):
        body = tk.Frame(self.root, bg=T["bg"])
        body.pack(fill=tk.BOTH, expand=True, padx=5, pady=(2, 0))
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=3)
        body.columnconfigure(2, weight=2)
        body.rowconfigure(0, weight=1)

        # Left: Original
        lf_l = self._lframe(body, "Original")
        lf_l.grid(row=0, column=0, sticky="nsew", padx=(0, 3))
        self.cv_orig = tk.Canvas(lf_l, bg=T["canvas"], highlightthickness=0)
        self.cv_orig.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Mid: Filtered
        lf_m = self._lframe(body, "Filtered")
        lf_m.grid(row=0, column=1, sticky="nsew", padx=3)
        self.cv_filt = tk.Canvas(lf_m, bg=T["canvas"], highlightthickness=0)
        self.cv_filt.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        self._lbl_filter = lf_m.winfo_children()[0]

        # Right panel
        right = tk.Frame(body, bg=T["bg"])
        right.grid(row=0, column=2, sticky="nsew", padx=(3, 0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=2)
        right.columnconfigure(0, weight=1)

        lf_info = self._lframe(right, "Analysis")
        lf_info.grid(row=0, column=0, sticky="nsew", pady=(0, 3))
        self.info_box = tk.Text(lf_info, font=("Menlo", 8), state=tk.DISABLED,
                                wrap=tk.WORD, bg=T["panel2"], fg=T["fg"],
                                relief=tk.FLAT, bd=0, padx=8, pady=6)
        self.info_box.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # Notebook
        self._nb = ttk.Notebook(right)
        self._nb.grid(row=1, column=0, sticky="nsew")

        tab_prof = tk.Frame(self._nb, bg=T["panel"])
        tab_cmd  = tk.Frame(self._nb, bg=T["panel"])
        self._nb.add(tab_prof, text="  Profiles  ")
        self._nb.add(tab_cmd,  text="  CMD Plot  ")

        self._build_profiles_tab(tab_prof)
        self._build_cmd_tab(tab_cmd)

        # Canvas bindings
        self.cv_filt.bind("<ButtonPress-1>",   self._on_press)
        self.cv_filt.bind("<B1-Motion>",        self._on_drag)
        self.cv_filt.bind("<ButtonRelease-1>",  self._on_release)
        self.cv_filt.bind("<Motion>",           self._on_hover)
        for cv in (self.cv_filt, self.cv_orig):
            cv.bind("<MouseWheel>", self._on_zoom)

        sb = tk.Frame(self.root, bg=T["button"], height=20)
        sb.pack(fill=tk.X)
        sb.pack_propagate(False)
        self.lbl_cursor = tk.Label(sb, text="  x=—  y=—", anchor="w",
                                   bg=T["button"], fg=T["button_fg"],
                                   font=("Menlo", 7))
        self.lbl_cursor.pack(side=tk.LEFT)
        self.lbl_zoom = tk.Label(sb, text="1.00x  ", anchor="e",
                                 bg=T["button"], fg=T["button_fg"],
                                 font=("Menlo", 7))
        self.lbl_zoom.pack(side=tk.RIGHT)

    # ── Profiles tab ─────────────────────────────────────────────────────────

    def _build_profiles_tab(self, parent):
        self._fig_prof = Figure(figsize=(5, 5.5), facecolor=T["bg"])
        self._fig_prof.subplots_adjust(hspace=0.65, wspace=0.45,
                                       top=0.92, bottom=0.09,
                                       left=0.14, right=0.97)
        self._ax_outer  = self._fig_prof.add_subplot(2, 2, 1)
        self._ax_inner  = self._fig_prof.add_subplot(2, 2, 2)
        self._ax_radial = self._fig_prof.add_subplot(2, 2, 3)
        self._ax_hist   = self._fig_prof.add_subplot(2, 2, 4)
        for ax, t, xl, yl in [
            (self._ax_outer,  "Outer Edge",    "dr (px)", "Intensity"),
            (self._ax_inner,  "Inner Edge",    "dr (px)", "Intensity"),
            (self._ax_radial, "Radial Profile","r (px)",  "Intensity"),
            (self._ax_hist,   "Histogram",     "I",       "Count"),
        ]:
            dax(ax, t, xl, yl)
        self._cv_prof = FigureCanvasTkAgg(self._fig_prof, master=parent)
        self._cv_prof.draw()
        self._cv_prof.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                           padx=2, pady=2)

    # ── CMD tab ───────────────────────────────────────────────────────────────

    def _build_cmd_tab(self, parent):
        bar = tk.Frame(parent, bg=T["panel2"], height=28)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        def lbl(t):
            tk.Label(bar, text=t, bg=T["panel2"], fg=T["fg2"],
                     font=("Helvetica", 7)).pack(side=tk.LEFT, padx=(6, 1), pady=5)

        lbl("Tip mT:")
        tk.Spinbox(bar, from_=-30, to=30, increment=0.01, format="%.3f",
                   textvariable=self._cmd_tip, width=7,
                   font=("Menlo", 7), bg=T["panel"], fg=T["fg"],
                   buttonbackground=T["panel"], relief=tk.FLAT, bd=0,
                   highlightthickness=1, highlightbackground=T["border"]
                   ).pack(side=tk.LEFT, pady=3)
        self._hbtn(bar, "Set",      self._cmd_redraw, "warning").pack(side=tk.LEFT, padx=3, pady=3)
        tk.Frame(bar, bg=T["border"], width=1).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=3)
        lbl("Style:")
        cb2 = ttk.Combobox(bar, textvariable=self._cmd_style,
                           state="readonly", values=["Dark", "White"],
                           width=7, font=("Helvetica", 7))
        cb2.pack(side=tk.LEFT, pady=3)
        cb2.bind("<<ComboboxSelected>>", lambda _: self._cmd_redraw())
        self._hbtn(bar, "Save PNG", self._cmd_save, "tool").pack(side=tk.RIGHT, padx=6, pady=3)

        # 3-panel figure
        self._fig_cmd = Figure(figsize=(5, 5.2), facecolor=T["bg"])
        gs = mgridspec.GridSpec(1, 3, figure=self._fig_cmd,
                                width_ratios=[3, 1.2, 1.2],
                                left=0.10, right=0.97,
                                top=0.88, bottom=0.12, wspace=0.05)
        self._ax_c1 = self._fig_cmd.add_subplot(gs[0])
        self._ax_c2 = self._fig_cmd.add_subplot(gs[1], sharey=self._ax_c1)
        self._ax_c3 = self._fig_cmd.add_subplot(gs[2], sharey=self._ax_c1)

        self._cv_cmd = FigureCanvasTkAgg(self._fig_cmd, master=parent)
        self._cv_cmd.draw()
        self._cv_cmd.get_tk_widget().pack(fill=tk.BOTH, expand=True,
                                          padx=2, pady=(0, 2))
        self._cv_cmd.mpl_connect("motion_notify_event", self._cmd_hover)

        sb = tk.Frame(parent, bg=T["button"], height=18)
        sb.pack(fill=tk.X)
        sb.pack_propagate(False)
        self._lbl_cmd_info = tk.Label(sb,
                                      text="  Upload image  ▶ Detect Stars",
                                      bg=T["button"], fg=T["fg2"],
                                      font=("Menlo", 7), anchor="w")
        self._lbl_cmd_info.pack(side=tk.LEFT)

        self._cmd_welcome()

    def _style_ttk(self):
        s = ttk.Style()
        s.theme_use("clam")
        s.configure("TCombobox",
                    fieldbackground=T["panel2"], background=T["panel2"],
                    foreground=T["fg"], selectbackground=T["panel2"],
                    selectforeground=T["fg"], arrowcolor=T["fg2"])
        s.configure("TNotebook", background=T["bg"], borderwidth=0, tabmargins=0)
        s.configure("TNotebook.Tab",
                    background=T["panel"], foreground=T["fg2"],
                    padding=(10, 4), font=("Helvetica", 9), borderwidth=0)
        s.map("TNotebook.Tab",
              background=[("selected", T["tab_sel"])],
              foreground=[("selected", T["fg"])])
        self._hi_tool("measure")

    # ══════════════════════════════════════════════════════════════════════════
    #  Widget helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _lframe(self, parent, title: str):
        f = tk.Frame(parent, bg=T["panel"],
                     highlightbackground=T["border"], highlightthickness=1)
        tk.Label(f, text=title.upper(), bg=T["panel"], fg=T["fg2"],
                 font=("TH Sarabun New", 16, "bold"), padx=6, pady=3
                 ).pack(anchor="w")
        return f

    def _hbtn(self, parent, text: str, cmd, style: str = "tool"):
        c = {
            "accent":  (T["accent"],  T["bg"]),
            "accent2": (T["accent2"], T["bg"]),
            "success": (T["success"], T["bg"]),
            "warning": (T["warning"], T["bg"]),
            "danger":  (T["danger"],  T["bg"]),
            "tool":    (T["panel2"],  T["fg"]),
        }
        bg, fg = c.get(style, (T["panel2"], T["fg"]))
        return tk.Button(parent, text=text, command=cmd,
                         bg=bg, fg=fg,
                         activebackground=T["border"],
                         activeforeground=T["fg"],
                         font=("Helvetica", 9), relief=tk.FLAT,
                         cursor="hand2", padx=9, pady=3, bd=0,
                         highlightthickness=0)

    # ══════════════════════════════════════════════════════════════════════════
    #  Load / Filter
    # ══════════════════════════════════════════════════════════════════════════

    def load_image(self):
        p = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("All", "*.*")])
        if not p:
            return
        raw = cv2.imread(p)
        if raw is None:
            messagebox.showerror("Error", "Cannot open file")
            return
        self.img_orig = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        self._reset()
        self.refresh_filter()

    def refresh_filter(self):
        if self.img_orig is None:
            return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        self.img_filtered = apply_filter(gray, self.filter_var.get())
        h, w = self.img_filtered.shape
        self._lbl_filter.config(
            text=f"FILTERED  {self.filter_var.get()}  {w}×{h}")
        self._cmd_cat = None
        self._cmd_welcome()
        self._redraw()

    def _reset(self):
        self.outer_circle = self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self.zoom, self.offset = 1.0, [0, 0]
        self._cmd_cat = None
        self._clear_profiles()
        self._cmd_welcome()
        self._status("Image loaded", T["fg2"])

    # ══════════════════════════════════════════════════════════════════════════
    #  Circle detection
    # ══════════════════════════════════════════════════════════════════════════

    def detect_circles(self):
        if self.img_orig is None:
            messagebox.showwarning("", "Open an image first")
            return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (9, 9), 2)
        h, w = gray.shape
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT,
            dp=1.2, minDist=max(20, min(h, w) // 8),
            param1=60, param2=30,
            minRadius=max(8, min(h, w) // 30),
            maxRadius=min(h, w) // 2)
        if circles is None:
            messagebox.showinfo("", "No circles detected")
            return
        circles = np.round(circles[0]).astype(int)
        circles = circles[circles[:, 2].argsort()[::-1]]
        self.outer_circle = tuple(circles[0])
        self.inner_circle = tuple(circles[1]) if len(circles) >= 2 else None
        self._log_circles()
        self._update_basic_plots()
        self._redraw()
        self._status(
            f"Outer r={circles[0][2]}px" +
            (f"  Inner r={circles[1][2]}px" if len(circles) >= 2 else ""),
            T["success"])

    def sobel_fit(self):
        if self.img_orig is None:
            return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 1.5)
        sx   = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        sy   = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        mag  = np.sqrt(sx**2 + sy**2)
        for thr, hi, attr in [
            (np.percentile(mag, 97), None,                   "outer_circle"),
            (np.percentile(mag, 94), np.percentile(mag, 97), "inner_circle"),
        ]:
            mask = ((mag >= thr) if hi is None
                    else ((mag >= thr) & (mag < hi))).astype(np.uint8)
            ys2, xs2 = np.where(mask > 0)
            if len(xs2) < 6:
                continue
            try:
                cx2, cy2, r2 = fit_circle_taubin(xs2, ys2)
                setattr(self, attr,
                        (int(round(cx2)), int(round(cy2)), int(round(r2))))
            except Exception:
                pass
        self._log_circles()
        self._update_basic_plots()
        self._redraw()

    # ══════════════════════════════════════════════════════════════════════════
    #  Eclipse analysis  (async)
    # ══════════════════════════════════════════════════════════════════════════

    def eclipse_analyze(self):
        if not self.eclipse_paths and self.img_orig is None:
            messagebox.showwarning("", "Open eclipse image first")
            return
        self._status("Analyzing… (background thread)", T["warning"])
        self._run_async(self._eclipse_work, (), self._eclipse_done)

    def _eclipse_work(self):
        """Heavy eclipse processing — runs in worker thread."""
        u = self.limb_u.get() / 100.0
        if len(self.eclipse_paths) > 1:
            img, info = align_and_median_combine(self.eclipse_paths)
            if img is None:
                raise RuntimeError("Alignment failed")
            cx, cy, r_moon = info
        else:
            img = (cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2BGR)
                   if self.img_orig is not None
                   else cv2.imread(self.eclipse_paths[0]))
            if img is None:
                raise RuntimeError("Cannot load image")
            g = cv2.GaussianBlur(
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, 1.2, 100,
                                  param1=100, param2=50,
                                  minRadius=50,
                                  maxRadius=int(min(img.shape[:2]) / 2))
            if c is None:
                raise RuntimeError("Moon not detected")
            x = np.uint16(np.around(c))[0, 0]
            cx, cy, r_moon = int(x[0]), int(x[1]), int(x[2])
        return detect_rings(img, cx, cy, r_moon, u)

    def _eclipse_done(self, res, exc):
        """Called on main thread after eclipse work completes."""
        if exc:
            messagebox.showerror("Error", str(exc))
            self._status("Analysis failed", T["danger"])
            return
        if res is None:
            messagebox.showerror("Error", "No umbra edge found")
            self._status("Analysis failed", T["danger"])
            return
        self.eclipse = res
        if res["o_r"] > 0:
            self.outer_circle = (int(res["o_cx"]), int(res["o_cy"]), int(res["o_r"]))
        if res["i_r"] > 0:
            self.inner_circle = (int(res["i_cx"]), int(res["i_cy"]), int(res["i_r"]))
        self._log_eclipse()
        self._update_eclipse_plots()
        self._redraw()
        self._status(
            f"Outer {res['o_r_km']:,.0f} km  "
            f"Inner {res['i_r_km']:,.0f} km  "
            f"err {res['err_r']:.2f}%",
            T["success"])

    # ══════════════════════════════════════════════════════════════════════════
    #  CMD — Star Detection  (async)
    # ══════════════════════════════════════════════════════════════════════════

    def run_cmd(self):
        if self.img_filtered is None:
            messagebox.showwarning("", "Upload image  ▶ Detect Stars")
            return
        self._status("Detecting stars… (background thread)", T["accent2"])
        self._run_async(self._cmd_work, (), self._cmd_done)

    def _cmd_work(self):
        """Heavy star detection — runs in worker thread."""
        gray = self.img_filtered
        bkg  = phot_background(gray, box=max(32, min(gray.shape) // 16))
        cat  = phot_detect(gray, bkg,
                           thresh_sigma=self._cmd_thresh.get(),
                           ap_r=self._cmd_ap.get())
        if cat is None:
            raise RuntimeError("ตรวจไม่พบดาว — ลด threshold σ หรือเพิ่ม aperture")
        return cat

    def _cmd_done(self, cat, exc):
        """Called on main thread after star detection completes."""
        if exc:
            messagebox.showerror("Detection Error", str(exc))
            self._status("Detection failed", T["danger"])
            return
        self._cmd_cat = cat
        self._cmd_tip.set(round(float(np.percentile(cat["mag"], 3)), 3))
        self._cmd_draw()
        self._nb.select(1)
        self._status(f"Found {len(cat['x']):,} stars from filtered image",
                     T["success"])
        self._lbl_cmd_info.config(
            text=(f"  N={len(cat['x']):,} stars  |  "
                  f"filter={self.filter_var.get()}  "
                  f"σ={self._cmd_thresh.get():.1f}  "
                  f"ap={self._cmd_ap.get()}px"),
            fg=T["success"])
        self._redraw()

    # ══════════════════════════════════════════════════════════════════════════
    #  CMD drawing
    # ══════════════════════════════════════════════════════════════════════════

    def _cmd_welcome(self):
        for ax in (self._ax_c1, self._ax_c2, self._ax_c3):
            ax.cla()
            ax.set_facecolor(T["plot_bg"])
            for sp in ax.spines.values():
                sp.set_color(T["border"])
            ax.set_xticks([])
            ax.set_yticks([])
        self._ax_c1.text(
            0.5, 0.5,
            "Upload image  ▶ Detect Stars\n\n"
            "Graph will use the Filtered image\n"
            "that is selected on the main page\n\n"
            "X = Concentration Index (CI)\n"
            "Y = Instrumental Magnitude",
            transform=self._ax_c1.transAxes,
            ha="center", va="center",
            color=T["fg"], fontsize=10,
            bbox=dict(boxstyle="round,pad=0.8",
                      fc=T["panel2"], ec=T["border"], alpha=0.92))
        self._fig_cmd.suptitle("CMD — Color-Magnitude Diagram",
                               color=T["ax_title"], fontsize=10,
                               fontweight="bold")
        self._cv_cmd.draw_idle()

    def _cmd_redraw(self):
        if self._cmd_cat is not None:
            self._cmd_draw()

    def _cmd_draw(self):
        cat = self._cmd_cat
        if cat is None:
            return

        pub     = self._cmd_style.get() == "White"
        bg_c    = "white"    if pub else T["plot_bg"]
        fg_c    = "black"    if pub else T["ax_title"]   # brighter title/label
        lbl_c   = "#333333"  if pub else T["ax_label"]   # axis labels
        tick_c  = "black"    if pub else T["ax_tick"]    # tick labels
        grid_c  = "#cccccc"  if pub else T["ax_grid"]
        sp_c    = "black"    if pub else T["border"]
        red_c   = "darkred"  if pub else "#ef4444"
        gray_c  = "gray"     if pub else "#9ca3af"
        fig_bg  = "white"    if pub else T["bg"]

        self._fig_cmd.set_facecolor(fig_bg)

        for ax in (self._ax_c1, self._ax_c2, self._ax_c3):
            ax.cla()
            ax.set_facecolor(bg_c)
            for sp in ax.spines.values():
                sp.set_color(sp_c)
                sp.set_linewidth(0.8)
            ax.tick_params(colors=tick_c, labelsize=7.5, length=3, width=0.8)
            ax.grid(color=grid_c, lw=0.5, ls="--", alpha=0.6)

        mag = cat["mag"]
        ci  = cat["ci"]
        tip = self._cmd_tip.get()

        # ── Panel 1: Scatter CMD ─────────────────────────────────────────────
        ax = self._ax_c1
        try:
            from scipy.stats import gaussian_kde
            rng = np.random.default_rng(0)
            n   = min(5000, len(mag))
            idx = rng.choice(len(mag), size=n, replace=False)
            kde = gaussian_kde(np.vstack([ci[idx], mag[idx]]), bw_method=0.07)
            dens = kde(np.vstack([ci[idx], mag[idx]]))
        except Exception:
            idx  = np.arange(min(5000, len(mag)))
            dens = np.ones(len(idx))

        bg_mask = np.ones(len(mag), bool)
        bg_mask[idx] = False
        ax.scatter(ci[bg_mask], mag[bg_mask], s=0.4,
                   c="#333355" if not pub else "#d0d0d0",
                   alpha=0.25, rasterized=True)
        ax.scatter(ci[idx], mag[idx], s=1.0,
                   c=dens,
                   cmap="inferno" if not pub else "Blues",
                   alpha=0.85, rasterized=True, linewidths=0)

        if tip != 0:
            ax.axhline(tip, color=gray_c, lw=1.4)
            Ir = np.percentile(mag, [1, 99])
            ax.text(0.02, tip - (Ir[1] - Ir[0]) * 0.015,
                    f"mT = {tip:.3f}",
                    color=fg_c, fontsize=9, fontweight="bold",
                    transform=ax.get_yaxis_transform())

        Ip = np.percentile(mag, [1, 99])
        ax.set_ylim(Ip[1] + 0.4, Ip[0] - 0.4)
        ax.set_xlim(-0.05, 1.1)
        ax.set_xlabel("Concentration Index (CI)", color=lbl_c, fontsize=8.5)
        ax.set_ylabel("Instrumental Mag",          color=lbl_c, fontsize=8.5)
        ax.set_title(self.filter_var.get(),         color=fg_c,  fontsize=8, pad=4)
        ax.text(0.02, 0.99, f"N = {len(mag):,}",
                transform=ax.transAxes, color=fg_c,
                fontsize=9.5, fontweight="bold", va="top")

        # ── Panel 2: CI profile per mag bin ─────────────────────────────────
        ax2   = self._ax_c2
        n_bins = max(20, len(mag) // 150)
        bins_m = np.linspace(Ip[0] - 0.2, Ip[1] + 0.2, n_bins)
        mid_m  = (bins_m[:-1] + bins_m[1:]) / 2
        ci_med = np.full(len(mid_m), np.nan)
        ci_p16 = np.full(len(mid_m), np.nan)
        ci_p84 = np.full(len(mid_m), np.nan)
        for k, (lo, hi) in enumerate(zip(bins_m[:-1], bins_m[1:])):
            sel = (mag >= lo) & (mag < hi)
            if sel.sum() > 4:
                ci_med[k] = np.median(ci[sel])
                ci_p16[k] = np.percentile(ci[sel], 16)
                ci_p84[k] = np.percentile(ci[sel], 84)
        ok = ~np.isnan(ci_med)
        for i in np.where(ok)[0]:
            ax2.plot([ci_p16[i], ci_p84[i]], [mid_m[i], mid_m[i]],
                     color="#444466" if not pub else "#aaaaaa",
                     lw=0.8, alpha=0.5)
        ax2.plot(ci_med[ok], mid_m[ok], color=red_c, lw=2.2)
        if tip != 0:
            ax2.axhline(tip, color=gray_c, lw=1.4)
        ax2.set_xlabel("CI", color=lbl_c, fontsize=8.5)
        ax2.set_title("CI Profile", color=fg_c, fontsize=8, pad=4)
        ax2.yaxis.set_tick_params(labelleft=False)
        ax2.spines["left"].set_visible(False)

        # ── Panel 3: Luminosity Function ─────────────────────────────────────
        ax3 = self._ax_c3
        counts, edges = np.histogram(mag, bins=max(30, len(mag) // 100))
        mid_lf = (edges[:-1] + edges[1:]) / 2
        sm = gaussian_filter1d(counts.astype(float), sigma=1.5)
        ax3.plot(sm, mid_lf, color=fg_c, lw=1.6, drawstyle="steps-mid")
        ax3.fill_betweenx(mid_lf, 0, sm, alpha=0.18,
                          color="black" if pub else T["fg"])
        if tip != 0:
            ax3.axhline(tip, color=gray_c, lw=1.4)
        ax3.invert_xaxis()
        ax3.set_xlabel("N",            color=lbl_c, fontsize=8.5)
        ax3.set_title("Lum. Function", color=fg_c,  fontsize=8, pad=4)
        ax3.yaxis.set_tick_params(labelleft=False)
        ax3.spines["left"].set_visible(False)

        self._fig_cmd.suptitle(
            f"CMD  —  {self.filter_var.get()}  —  N={len(mag):,} stars",
            color=fg_c, fontsize=10, fontweight="bold", y=0.98)
        self._cv_cmd.draw_idle()

    def _cmd_hover(self, event):
        if event.inaxes is self._ax_c1 and event.xdata and event.ydata:
            self._lbl_cmd_info.config(
                text=f"  CI={event.xdata:.3f}  mag={event.ydata:.3f}")

    def _cmd_save(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if not path:
            return
        pub = self._cmd_style.get() == "White"
        self._fig_cmd.savefig(path,
                              dpi=200 if pub else 150,
                              facecolor=self._fig_cmd.get_facecolor(),
                              bbox_inches="tight")
        self._lbl_cmd_info.config(
            text=f"  Saved: {path.split('/')[-1]}", fg=T["success"])

    # ══════════════════════════════════════════════════════════════════════════
    #  Profile plots
    # ══════════════════════════════════════════════════════════════════════════

    def _clear_profiles(self):
        for ax, t, xl, yl in [
            (self._ax_outer,  "Outer Edge",    "dr (px)", "Intensity"),
            (self._ax_inner,  "Inner Edge",    "dr (px)", "Intensity"),
            (self._ax_radial, "Radial Profile","r (px)",  "Intensity"),
            (self._ax_hist,   "Histogram",     "I",       "Count"),
        ]:
            ax.cla()
            dax(ax, t, xl, yl)
        self._cv_prof.draw_idle()

    def _radial_profile_for(self, circle, gray):
        cx, cy, r = circle
        h, w = gray.shape
        sw = int(r * 0.45)
        dr = np.arange(-sw, sw + 1, dtype=float)
        profiles = np.full((180, len(dr)), np.nan)
        for i in range(180):
            ca = math.cos(math.pi * i / 180)
            sa = math.sin(math.pi * i / 180)
            for j, d in enumerate(dr):
                px = int(round(cx + (r + d) * ca))
                py = int(round(cy + (r + d) * sa))
                if 0 <= px < w and 0 <= py < h:
                    profiles[i, j] = gray[py, px]
        mp = np.nanmean(profiles, axis=0)
        sp = np.nanstd(profiles, axis=0)
        return dr, mp, sp, np.gradient(mp)

    def _update_basic_plots(self):
        if self.img_orig is None:
            return
        gray = cv2.cvtColor(self.img_orig, cv2.COLOR_RGB2GRAY)

        for circle, ax, col, lbl in [
            (self.outer_circle, self._ax_outer, T["outer"], "Outer"),
            (self.inner_circle, self._ax_inner, T["inner"], "Inner"),
        ]:
            ax.cla()
            dax(ax, f"{lbl} Edge", "dr (px)", "Intensity")
            if circle is None:
                continue
            dr, mp, sp, gp = self._radial_profile_for(circle, gray)
            ei = int(np.argmin(gp))
            ax.plot(dr, mp,   color=col,       lw=1.4, label="Mean")
            ax.fill_between(dr, mp - sp, mp + sp, alpha=0.15, color=col)
            ax.plot(dr, gp * 3, color="#c0c8e8", lw=0.9, ls="--", label="Grad×3")
            ax.axvline(dr[ei], color=T["warning"], lw=1.3, ls=":",
                       label=f"{dr[ei]:+.1f}px")
            styled_legend(ax, fontsize=7, loc="upper right")

        # Radial profile
        self._ax_radial.cla()
        dax(self._ax_radial, "Radial Profile", "r (px)", "Intensity")
        for circle, col, lbl in [
            (self.outer_circle, T["outer"], "outer"),
            (self.inner_circle, T["inner"], "inner"),
        ]:
            if circle is None:
                continue
            cx, cy, r = circle
            rv = np.arange(0, int(r * 1.5))
            profs = []
            for i in range(0, 360, 3):
                ca = math.cos(math.pi * i / 180)
                sa = math.sin(math.pi * i / 180)
                xi = np.clip((cx + rv * ca).astype(int), 0, gray.shape[1] - 1)
                yi = np.clip((cy + rv * sa).astype(int), 0, gray.shape[0] - 1)
                profs.append(gray[yi, xi].astype(float))
            mp2 = np.mean(profs, axis=0)
            self._ax_radial.plot(rv, mp2, color=col, lw=1.3, label=lbl)
            self._ax_radial.axvline(r, color=col, lw=1, ls="--", alpha=0.5)
        styled_legend(self._ax_radial, fontsize=7)

        # Histogram
        self._ax_hist.cla()
        dax(self._ax_hist, "Histogram", "Intensity", "Count")
        if self.img_filtered is not None:
            self._ax_hist.hist(self.img_filtered.flatten(),
                               bins=64, range=(0, 256),
                               color=T["accent2"], alpha=0.85, edgecolor="none")

        self._cv_prof.draw_idle()

    def _update_eclipse_plots(self):
        r = self.eclipse
        if r is None:
            return
        gray = cv2.cvtColor(r["img_bgr"], cv2.COLOR_BGR2GRAY)

        # Outer edge
        self._ax_outer.cla()
        dax(self._ax_outer, "Outer Edge (Umbra)", "dr (px)", "Intensity")
        if self.outer_circle:
            dr, mp, sp, gp = self._radial_profile_for(self.outer_circle, gray)
            ei = int(np.argmin(gp))
            self._ax_outer.plot(dr, mp, color=T["outer"], lw=1.4, label="Mean")
            self._ax_outer.fill_between(dr, mp - sp, mp + sp,
                                        alpha=0.15, color=T["outer"])
            self._ax_outer.plot(dr, gp * 3, color="#c0c8e8", lw=0.9,
                                ls="--", label="Grad×3")
            self._ax_outer.axvline(dr[ei], color=T["warning"], lw=1.3, ls=":",
                                   label=f"edge {dr[ei]:+.1f}px")
            styled_legend(self._ax_outer, fontsize=7)

        # Inner edge
        self._ax_inner.cla()
        dax(self._ax_inner, "Inner Edge (Moon limb)", "dr (px)", "Intensity")
        if self.inner_circle:
            dr2, mp2, sp2, gp2 = self._radial_profile_for(self.inner_circle, gray)
            ei2 = int(np.argmin(gp2))
            self._ax_inner.plot(dr2, mp2, color=T["inner"], lw=1.4, label="Mean")
            self._ax_inner.fill_between(dr2, mp2 - sp2, mp2 + sp2,
                                        alpha=0.15, color=T["inner"])
            self._ax_inner.plot(dr2, gp2 * 3, color="#c0c8e8", lw=0.9,
                                ls="--", label="Grad×3")
            self._ax_inner.axvline(dr2[ei2], color=T["warning"], lw=1.3, ls=":",
                                   label=f"edge {dr2[ei2]:+.1f}px")
            styled_legend(self._ax_inner, fontsize=7)

        # Full radial
        self._ax_radial.cla()
        dax(self._ax_radial, "Full Radial", "r (px)", "Intensity")
        cx, cy, rm = r["cx"], r["cy"], r["r_moon"]
        rv2 = np.arange(0, int(1.8 * rm))
        profs = []
        for i in range(0, 360, 3):
            ca = math.cos(math.pi * i / 180)
            sa = math.sin(math.pi * i / 180)
            xi = np.clip((cx + rv2 * ca).astype(int), 0, gray.shape[1] - 1)
            yi = np.clip((cy + rv2 * sa).astype(int), 0, gray.shape[0] - 1)
            profs.append(gray[yi, xi].astype(float))
        mp3 = np.mean(profs, axis=0)
        sp3 = np.std(profs, axis=0)
        self._ax_radial.plot(rv2, mp3, color=T["accent"], lw=1.4)
        self._ax_radial.fill_between(rv2, mp3 - sp3, mp3 + sp3,
                                     alpha=0.1, color=T["accent"])
        self._ax_radial.axvline(rm, color=T["inner"], lw=1.1, ls="--",
                                label=f"Moon r={rm}px")
        if r["o_r"] > 0:
            self._ax_radial.axvline(r["o_r"], color=T["outer"], lw=1.1, ls="--",
                                    label=f"Umbra r={r['o_r']:.0f}px")
        styled_legend(self._ax_radial, fontsize=7)

        # Histogram
        self._ax_hist.cla()
        dax(self._ax_hist, "Histogram", "Intensity", "Count")
        gf = gray.flatten()
        self._ax_hist.hist(gf, bins=64, range=(0, 256),
                           color=T["accent2"], alpha=0.85, edgecolor="none")
        self._ax_hist.axvline(np.percentile(gf, 95) * 0.35,
                              color=T["danger"], lw=1.2, ls="--",
                              label="shadow thresh")
        styled_legend(self._ax_hist, fontsize=7)

        self._cv_prof.draw_idle()

    # ══════════════════════════════════════════════════════════════════════════
    #  Canvas render
    # ══════════════════════════════════════════════════════════════════════════

    def _redraw(self):
        if not self._pending:
            self._pending = True
            self.root.after(16, self._do_redraw)

    def _do_redraw(self):
        self._pending = False
        if self.img_filtered is None:
            return
        self._draw_canvas(self.cv_orig, self._make_orig())
        self._draw_canvas(self.cv_filt, self._make_filt())

    def _scale(self, canvas, iw, ih):
        cw = canvas.winfo_width() or 500
        ch = canvas.winfo_height() or 400
        sc = min(cw / iw, ch / ih) * 0.92 * self.zoom
        return sc, cw // 2 + self.offset[0] - int(iw * sc) // 2, \
                   ch // 2 + self.offset[1] - int(ih * sc) // 2

    def _make_orig(self):
        h, w = self.img_orig.shape[:2]
        sc, ox, oy = self._scale(self.cv_orig, w, h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        pil = Image.fromarray(self.img_orig).resize((nw, nh), Image.BILINEAR)
        self._draw_rings(ImageDraw.Draw(pil), sc)
        return pil

    def _make_filt(self):
        h, w = self.img_filtered.shape
        sc, ox, oy = self._scale(self.cv_filt, w, h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        pil  = Image.fromarray(self.img_filtered).resize((nw, nh), Image.BILINEAR)
        draw = ImageDraw.Draw(pil)

        # Star markers
        if self._cmd_cat is not None:
            for sx, sy in zip(self._cmd_cat["x"], self._cmd_cat["y"]):
                px, py = int(sx * sc), int(sy * sc)
                draw.ellipse([px - 3, py - 3, px + 3, py + 3],
                             outline=T["accent2"], width=1)

        # Measurement
        if self.measure:
            p1, p2, d = self.measure
            s1 = (int(p1[0] * sc), int(p1[1] * sc))
            s2 = (int(p2[0] * sc), int(p2[1] * sc))
            draw.line([s1, s2], fill="#ffffff", width=2)
            for pt in (s1, s2):
                draw.ellipse([pt[0] - 4, pt[1] - 4,
                              pt[0] + 4, pt[1] + 4], fill=T["danger"])
            km = (f"  {d * self.eclipse['km_pp']:.0f}km"
                  if self.eclipse else "")
            draw.text(((s1[0] + s2[0]) // 2 + 4, (s1[1] + s2[1]) // 2 - 12),
                      f"{d:.1f}px{km}", fill=T["warning"])

        # ROI
        if self.roi:
            x1, y1, x2, y2 = self.roi
            draw.rectangle([int(x1 * sc), int(y1 * sc),
                            int(x2 * sc), int(y2 * sc)],
                           outline=T["success"], width=2)

        self._draw_rings(draw, sc)

        # Live drawing preview
        if self._drawing and self._pt_start and self._pt_end:
            s1 = (int(self._pt_start[0] * sc), int(self._pt_start[1] * sc))
            s2 = (int(self._pt_end[0]   * sc), int(self._pt_end[1]   * sc))
            if self.tool == "roi":
                draw.rectangle([s1, s2], outline=T["warning"], width=1)
            elif self.tool == "measure":
                draw.line([s1, s2], fill=T["accent"], width=1)

        return pil

    def _draw_rings(self, draw, sc):
        for circle, col, lbl in [
            (self.outer_circle, T["outer"], "outer"),
            (self.inner_circle, T["inner"], "inner"),
        ]:
            if circle is None:
                continue
            cx, cy, r = circle
            scx = int(cx * sc)
            scy = int(cy * sc)
            sr  = int(r  * sc)
            pts = [(scx + int(sr * math.cos(2 * math.pi * i / 360)),
                    scy + int(sr * math.sin(2 * math.pi * i / 360)))
                   for i in range(361)]
            draw.line(pts, fill=col, width=2)
            draw.line([(scx - 8, scy), (scx + 8, scy)], fill=col, width=1)
            draw.line([(scx, scy - 8), (scx, scy + 8)], fill=col, width=1)
            draw.text((scx + sr + 4, scy - 10), f"{lbl} r={r}", fill=col)

    def _draw_canvas(self, canvas, pil):
        h, w = (self.img_filtered.shape
                if canvas is self.cv_filt
                else self.img_orig.shape[:2])
        sc, ox, oy = self._scale(canvas, w, h)
        nw, nh = max(1, int(w * sc)), max(1, int(h * sc))
        tk_img = ImageTk.PhotoImage(pil)
        self._tk_imgs[id(canvas)] = tk_img
        canvas.delete("all")
        canvas.create_image(ox + nw // 2, oy + nh // 2, image=tk_img)
        canvas.create_line(ox, oy, ox + nw, oy,   fill=T["border"])
        canvas.create_line(ox, oy, ox,   oy + nh, fill=T["border"])
        step = max(50, int(100 / sc))
        for i in range(0, 9999, step):
            px = ox + int(i * sc)
            if px > ox + nw:
                break
            canvas.create_text(px, oy - 8, text=str(i),
                                fill=T["fg2"], font=("Helvetica", 6))
        for i in range(0, 9999, step):
            py = oy + int(i * sc)
            if py > oy + nh:
                break
            canvas.create_text(ox - 18, py, text=str(i),
                                fill=T["fg2"], font=("Helvetica", 6))

    # ══════════════════════════════════════════════════════════════════════════
    #  Events
    # ══════════════════════════════════════════════════════════════════════════

    def _img_coord(self, ex, ey):
        if self.img_filtered is None:
            return (0, 0)
        h, w = self.img_filtered.shape
        sc, ox, oy = self._scale(self.cv_filt, w, h)
        return (int((ex - ox) / sc), int((ey - oy) / sc))

    def _on_press(self, e):
        if self.img_filtered is None:
            return
        self._drawing   = True
        self._pt_start  = self._img_coord(e.x, e.y)
        self._pt_end    = self._pt_start
        if self.tool == "inspect":
            self._show_pixel(self._pt_start)
        elif self.tool == "pan":
            self._pan_start = (e.x - self.offset[0], e.y - self.offset[1])

    def _on_drag(self, e):
        if not self._drawing:
            return
        if self.tool == "pan":
            self.offset = [e.x - self._pan_start[0],
                           e.y - self._pan_start[1]]
            self._redraw()
            return
        self._pt_end = self._img_coord(e.x, e.y)
        if self.tool in ("measure", "roi"):
            self._redraw()

    def _on_release(self, e):
        if not self._drawing:
            return
        self._drawing = False
        self._pt_end  = self._img_coord(e.x, e.y)
        if self.tool == "measure" and self._pt_start:
            p1, p2 = self._pt_start, self._pt_end
            d = np.linalg.norm(np.array(p1) - np.array(p2))
            self.measure = (p1, p2, d)
            km = (f"\nDistance  {d * self.eclipse['km_pp']:,.1f} km"
                  if self.eclipse else "")
            self._log(f"Measurement\n{'-'*24}\nFrom  {p1}\nTo    {p2}"
                      f"\nPixels {d:.2f}{km}")
        elif self.tool == "roi" and self._pt_start:
            x1, y1 = self._pt_start
            x2, y2 = self._pt_end
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            hi, wi = self.img_filtered.shape
            self.roi = (max(0, x1), max(0, y1),
                        min(wi - 1, x2), min(hi - 1, y2))
            self._calc_roi()
        self._redraw()

    def _on_hover(self, e):
        if self.img_filtered is None:
            return
        x, y = self._img_coord(e.x, e.y)
        h, w = self.img_filtered.shape
        v = (f"  I={self.img_filtered[y, x]}"
             if 0 <= x < w and 0 <= y < h else "")
        self.lbl_cursor.config(text=f"  x={x}  y={y}{v}")

    def _on_zoom(self, e):
        self.zoom = max(0.05,
                        min(self.zoom * (1.12 if e.delta > 0 else 0.89), 40.0))
        self.lbl_zoom.config(text=f"{self.zoom:.2f}x  ")
        self._redraw()

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def set_tool(self, t: str):
        self.tool    = t
        self.measure = None
        self._hi_tool(t)
        cursor_map = {
            "measure": "crosshair",
            "roi":     "sizing",
            "inspect": "tcross",
            "pan":     "fleur",
        }
        self.cv_filt.config(cursor=cursor_map.get(t, "arrow"))
        self._redraw()

    def _hi_tool(self, active: str):
        for k, b in self._tool_btns.items():
            b.config(bg=T["accent"] if k == active else T["panel2"],
                     fg=T["bg"]     if k == active else T["fg"])

    def _clear_all(self):
        self.outer_circle = self.inner_circle = None
        self.roi = self.measure = self.eclipse = None
        self._cmd_cat = None
        self._clear_profiles()
        self._cmd_welcome()
        self._redraw()
        self._status("Cleared", T["fg2"])

    def _calc_roi(self):
        x1, y1, x2, y2 = self.roi
        crop = self.img_filtered[y1:y2, x1:x2]
        if crop.size == 0:
            return
        km = ""
        if self.eclipse:
            k  = self.eclipse["km_pp"]
            km = f"\nW {(x2-x1)*k:,.1f} km  H {(y2-y1)*k:,.1f} km"
        self._log(
            f"ROI\n{'-'*24}\n({x1},{y1})–({x2},{y2})\n"
            f"{x2-x1}×{y2-y1} px{km}\n"
            f"Mean {crop.mean():.1f}  Std {crop.std():.1f}\n"
            f"Range {crop.min()}–{crop.max()}")

    def _show_pixel(self, pt):
        x, y = pt
        h, w = self.img_filtered.shape
        if not (0 <= x < w and 0 <= y < h):
            return
        v  = self.img_filtered[y, x]
        km = (f"\nDist Moon "
              f"{math.hypot(x - self.eclipse['cx'], y - self.eclipse['cy']) * self.eclipse['km_pp']:,.1f} km"
              if self.eclipse else "")
        self._log(f"Pixel ({x}, {y})\n{'-'*24}\nI={v}  ({v/255:.3f}){km}")

    def _log(self, msg: str):
        self.info_box.config(state=tk.NORMAL)
        self.info_box.delete(1.0, tk.END)
        self.info_box.insert(tk.END, msg)
        self.info_box.config(state=tk.DISABLED)

    def _log_circles(self):
        lines = ["Circle Detection\n" + "-" * 24]
        for c, lbl in [(self.outer_circle, "Outer"),
                       (self.inner_circle, "Inner")]:
            if c:
                lines.append(
                    f"{lbl}\n"
                    f"  center ({c[0]},{c[1]})\n"
                    f"  r={c[2]}px\n"
                    f"  area {math.pi*c[2]**2:,.0f}px²")
        self._log("\n".join(lines))

    def _log_eclipse(self):
        r = self.eclipse
        self._log(
            f"Eclipse Analysis\n{'='*26}\n"
            f"km/pixel  {r['km_pp']:.4f}\n\n"
            f"OUTER  (Umbra)\n"
            f"  r  {r['o_r']:.1f}px  {r['o_r_km']:,.1f}km\n"
            f"  area  {r['o_area']:,.0f}km²\n\n"
            f"INNER  (Moon limb)\n"
            f"  r  {r['i_r']:.1f}px  {r['i_r_km']:,.1f}km\n\n"
            f"NASA theoretical\n"
            f"  r  {r['theo_r_km']:,.1f}km  area  {r['theo_area']:,.0f}km²\n\n"
            f"Radius error  {r['err_r']:.3f}%\n"
            f"Area error    {r['err_area']:.3f}%\n"
            f"Shadow cover  {r['dark_pct']:.2f}%\n"
            f"RANSAC  {'on' if HAS_SKIMAGE else 'off'}")

    def _status(self, msg: str, col=None):
        self.lbl_status.config(text=msg, fg=col or T["fg2"])


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()