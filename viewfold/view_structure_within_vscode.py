import os
import json
import uuid
import glob
import numpy as np
from ase.io import read
from scipy.sparse.csgraph import connected_components
from ase.geometry import find_mic
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from IPython.display import display, HTML

# =====================================================================
# 1. 查找并读取结构
# =====================================================================
base_dir = "/path/to/A02_cp-mace/Cu111_44_CO2"   # ← ←←←←←←←←←←←←←←←←←←←←←修改为你的实际路径

patterns = ["*.xsd", "POSCAR*", "CONTCAR*", "*.vasp", "*.cif",
            "*.xyz", "*.extxyz", "*.traj", "*.pdb"]

# 把晶胞外的原子放回晶胞：
#   "molecule" 小分子整体放回（保持分子完整），金属板等大块逐原子放回（推荐）
#   "atom"     每个原子单独放回（最整齐，但跨边界的分子会被拆开）import os
import json
import uuid
import glob
import numpy as np
from ase.io import read
from scipy.sparse.csgraph import connected_components
from ase.geometry import find_mic
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from ase.data import covalent_radii
from ase.data.colors import jmol_colors
from IPython.display import display, HTML

# =====================================================================
# 1. 查找并读取结构
# =====================================================================
base_dir = "/path/to/A02_cp-mace/Cu111_44_CO2"   # ← ←←←←←←←←←←←←←←←←←←←←←修改为你的实际路径

patterns = ["*.xsd", "POSCAR*", "CONTCAR*", "*.vasp", "*.cif",
            "*.xyz", "*.extxyz", "*.traj", "*.pdb"]

# 把晶胞外的原子放回晶胞：
#   "molecule" 小分子整体放回（保持分子完整），金属板等大块逐原子放回（推荐）
#   "atom"     每个原子单独放回（最整齐，但跨边界的分子会被拆开）
#   None       不处理，按文件原样显示
WRAP_MODE = "molecule"


def wrap_into_cell(atoms, mode="molecule", max_mol_size=20):
    """把原子放回晶胞内。
    mode="atom"     : 每个原子单独放回晶胞（最整齐，但跨边界的分子会被拆开）
    mode="molecule" : 小分子整体平移回晶胞（保持分子完整），大的部分（如金属板）逐原子放回
    """
    atoms = atoms.copy()
    if not atoms.pbc.any() or mode is None:
        return atoms
    if mode == "atom":
        atoms.wrap()
        return atoms

    nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
    n_comp, labels = connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)

    pos = atoms.positions.copy()
    cell, pbc = atoms.cell, atoms.pbc
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) > max_mol_size:                 # 金属板等大块：逐原子放回
            frac = cell.scaled_positions(pos[idx])
            frac[:, pbc] %= 1.0
            pos[idx] = cell.cartesian_positions(frac)
            continue
        # 小分子：先用最小镜像把分子拼完整，再把分子中心平移进晶胞
        ref = pos[idx[0]]
        d, _ = find_mic(pos[idx] - ref, cell, pbc)
        mol = ref + d
        center = mol.mean(axis=0)
        fc = cell.scaled_positions(center[None])[0]
        shift_frac = np.where(pbc, np.floor(fc), 0.0)
        pos[idx] = mol - cell.cartesian_positions(shift_frac[None])[0]
    atoms.positions = pos
    return atoms


files = []
for p in patterns:
    files += glob.glob(os.path.join(base_dir, "**", p), recursive=True)
files = sorted(set(files))

images, names = [], []
for f in files:
    try:
        images.append(wrap_into_cell(read(f, index=-1), WRAP_MODE))   # 多帧文件取最后一帧
        names.append(os.path.relpath(f, base_dir))
    except Exception as e:
        print(f"读取失败 {f}: {e}")

print(f"共读取 {len(images)} 个结构")

# =====================================================================
# 2. 查看器设置
# =====================================================================
CANVAS_W, CANVAS_H = 920, 580     # 默认画布大小（页面里也可以改）
SPHERE_QUALITY = "medium"         # 球的精细度：low / medium / high，越低越流畅
PLAY_INTERVAL_MS = 800            # 自动播放时每个结构停留的毫秒数


# =====================================================================
# 3. 生成整页查看器（所有结构一次性放进网页，切换在浏览器里完成）
# =====================================================================
def build_viewer(images, names, width=920, height=580, quality="medium", interval=800):
    uid = uuid.uuid4().hex[:8]

    # 每种元素只定义一次球体，其余原子复用（USE），减少内存和渲染开销
    elements = sorted({int(z) for a in images for z in a.numbers})
    defs = []
    for z in elements:
        r, g, b = jmol_colors[z]
        defs.append(
            f'<shape DEF="el{z}_{uid}"><appearance><material diffuseColor="{r:.3f} {g:.3f} {b:.3f}"></material>'
            f'</appearance><sphere radius="{covalent_radii[z]:.3f}"></sphere></shape>')

    def cell_lines(cell):
        o = np.zeros(3)
        a, b, c = cell
        corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
                 (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        pts = " ".join(f"{corners[i][0]:.3f} {corners[i][1]:.3f} {corners[i][2]:.3f} "
                       f"{corners[j][0]:.3f} {corners[j][1]:.3f} {corners[j][2]:.3f}"
                       for i, j in edges)
        return ('<shape><appearance><material emissiveColor="0 0 0" diffuseColor="0 0 0"></material></appearance>'
                f'<lineset vertexCount="{" ".join(["2"] * 12)}"><coordinate point="{pts}"></coordinate>'
                '</lineset></shape>')

    groups, meta = [], []
    all_lo, all_hi = np.full(3, np.inf), np.full(3, -np.inf)
    cell_lo, cell_hi = np.full(3, np.inf), np.full(3, -np.inf)
    for atoms, name in zip(images, names):
        parts = []
        if atoms.pbc.any():
            parts.append(cell_lines(atoms.cell))
            c = atoms.cell
            corners = np.array([i * c[0] + j * c[1] + k * c[2]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            cell_lo, cell_hi = np.minimum(cell_lo, corners.min(0)), np.maximum(cell_hi, corners.max(0))
        for (x, y, zc), z in zip(atoms.positions, atoms.numbers):
            parts.append(f'<transform translation="{x:.3f} {y:.3f} {zc:.3f}">'
                         f'<shape USE="el{int(z)}_{uid}"></shape></transform>')
        groups.append("<group>" + "".join(parts) + "</group>")
        all_lo = np.minimum(all_lo, atoms.positions.min(0))
        all_hi = np.maximum(all_hi, atoms.positions.max(0))
        meta.append(dict(name=name, formula=atoms.get_chemical_formula(), n=len(atoms)))

    if not np.isfinite(cell_lo).all():          # 没有晶胞时，"整个晶胞"模式退化为只看原子
        cell_lo, cell_hi = all_lo, all_hi

    cfg = dict(uid=uid, meta=meta, width=width, height=height, interval=interval,
               atoms_box=[all_lo.tolist(), all_hi.tolist()],
               cell_box=[cell_lo.tolist(), cell_hi.tolist()])

    html = HTML_TEMPLATE
    for key, val in {
        "__UID__": uid,
        "__W__": str(width), "__H__": str(height),
        "__QUALITY__": quality,
        "__DEFS__": "".join(defs),
        "__GROUPS__": "".join(groups),
        "__N__": str(len(images) - 1),
        "__CFG__": json.dumps(cfg),
    }.items():
        html = html.replace(key, val)
    return html


HTML_TEMPLATE = r"""
<link rel="stylesheet" type="text/css" href="https://www.x3dom.org/release/x3dom.css">
<script type="text/javascript" src="https://www.x3dom.org/release/x3dom.js"></script>
<style>
  #av___UID__ { background:#fff; color:#000; font:13px/1.4 sans-serif; padding:10px 12px;
                border-radius:6px; display:inline-block; outline:none; }
  #av___UID__ .row { display:flex; align-items:center; gap:8px; margin:4px 0; flex-wrap:wrap; }
  #av___UID__ button { padding:3px 10px; border:1px solid #bbb; background:#f3f3f3; border-radius:3px;
                       cursor:pointer; color:#000; }
  #av___UID__ button.on { background:#c8c8c8; }
  #av___UID__ select, #av___UID__ input[type=number] { color:#000; background:#fff; border:1px solid #bbb; }
  #av___UID__ input[type=number] { width:70px; }
  #av___UID__ .help { font-size:11px; color:#555; }
</style>

<div id="av___UID__" tabindex="0">
  <div class="row">
    <button data-act="play">▶</button>
    <button data-act="prev">◀</button>
    <input type="range" min="0" max="__N__" value="0" style="width:420px" data-el="slider">
    <button data-act="next">▶|</button>
    <span data-el="idx"></span>
  </div>
  <div class="row">
    视角
    <button data-view="0" class="on">正视 (沿y, z朝上)</button>
    <button data-view="1">侧视 (沿x, z朝上)</button>
    <button data-view="2">俯视 (沿z)</button>
    &nbsp; 框选
    <button data-fit="atoms" class="on">只看原子</button>
    <button data-fit="cell">整个晶胞</button>
  </div>
  <div class="row">
    画布宽 <input type="number" data-el="w" value="__W__" step="10" min="200">
    高 <input type="number" data-el="h" value="__H__" step="10" min="200">
    <button data-act="reset">复位视角 (0)</button>
    <span data-el="zoominfo" class="help"></span>
  </div>
  <div class="help">
    鼠标：Ctrl+滚轮 缩放 | Alt+左键拖动 平移 | 左键拖动 旋转 &nbsp;&nbsp;
    键盘（鼠标放在本区域内）：←/→ 切换结构 | ↑/↓ 上下平移 | Shift+←/→ 左右平移 | +/- 缩放 | 1/2/3 视角 | 0 复位 | 空格 播放
  </div>
  <div class="row"><b data-el="info"></b></div>
  <X3D id="x3d___UID__" width="__W__px" height="__H__px" disableKeys="true"
       style="border:1px solid #ccc; background:#fff;">
    <param name="PrimitiveQuality" value="__QUALITY__">
    <scene>
      <OrthoViewpoint id="vp___UID__" position="0 0 100" fieldOfView="-10 -10 10 10"></OrthoViewpoint>
      <background skyColor="1 1 1"></background>
      <switch whichChoice="-1">__DEFS__</switch>
      <switch id="sw___UID__" whichChoice="0">__GROUPS__</switch>
    </scene>
  </X3D>
</div>

<script>
(function () {
  var cfg = __CFG__;
  var root = document.getElementById("av_" + cfg.uid);
  var x3d = document.getElementById("x3d_" + cfg.uid);
  var q = function (sel) { return root.querySelector(sel); };
  var qa = function (sel) { return root.querySelectorAll(sel); };
  var slider = q('[data-el=slider]');

  var VIEWS = [
    {orient: [1, 0, 0, 1.5708],                 axis: 1, sign: -1, h: 0, v: 2},
    {orient: [0.57735, 0.57735, 0.57735, 2.0944], axis: 0, sign: +1, h: 1, v: 2},
    {orient: [0, 0, 1, 0],                      axis: 2, sign: +1, h: 0, v: 1}
  ];
  var st = {i: 0, view: 0, fit: "atoms", zoom: 1, dh: 0, dv: 0,
            W: cfg.width, H: cfg.height, playing: null};
  var vp, sw, F;

  // ---------- 相机 ----------
  function geom() {
    var box = st.fit === "cell" ? cfg.cell_box : cfg.atoms_box;
    var V = VIEWS[st.view], lo = box[0], hi = box[1];
    var eh = (hi[V.h] - lo[V.h]) * 1.1 + 2, ev = (hi[V.v] - lo[V.v]) * 1.1 + 2;
    var base = st.W <= st.H ? Math.max(eh / 2, ev / 2 * st.W / st.H)
                            : Math.max(ev / 2, eh / 2 * st.H / st.W);
    var c = [0, 1, 2].map(function (k) { return (lo[k] + hi[k]) / 2; });
    return {V: V, base: base, c: c};
  }
  function applyCamera(resetRotation) {
    if (!vp) return;
    var g = geom(), half = g.base / st.zoom;
    vp.setFieldValue("fieldOfView", new F.MFFloat([-half, -half, half, half]));
    if (resetRotation) {
      var c = g.c.slice();
      c[g.V.h] += st.dh; c[g.V.v] += st.dv;
      var p = c.slice();
      p[g.V.axis] += g.V.sign * (g.base * 10 + 50);
      var o = g.V.orient;
      vp.setFieldValue("orientation", F.Quaternion.axisAngle(new F.SFVec3f(o[0], o[1], o[2]), o[3]));
      vp.setFieldValue("centerOfRotation", new F.SFVec3f(c[0], c[1], c[2]));
      vp.setFieldValue("position", new F.SFVec3f(p[0], p[1], p[2]));
    }
    q('[data-el=zoominfo]').textContent = "缩放 " + st.zoom.toFixed(2) + "×";
  }

  // ---------- 结构切换（只改 Switch 节点，不重新加载，非常快） ----------
  function show(i) {
    var n = cfg.meta.length;
    st.i = Math.max(0, Math.min(n - 1, i));
    slider.value = st.i;
    if (sw) sw.setFieldValue("whichChoice", st.i);
    var m = cfg.meta[st.i];
    q('[data-el=idx]').textContent = st.i + " / " + (n - 1);
    q('[data-el=info]').textContent = "[" + st.i + "] " + m.name + "  |  " + m.formula + "  |  " + m.n + " 个原子";
  }
  function togglePlay() {
    var btn = q('[data-act=play]');
    if (st.playing) { clearInterval(st.playing); st.playing = null; btn.textContent = "▶"; return; }
    btn.textContent = "❚❚";
    st.playing = setInterval(function () {
      show(st.i >= cfg.meta.length - 1 ? 0 : st.i + 1);
    }, cfg.interval);
  }
  function setView(k) {
    st.view = k; st.dh = 0; st.dv = 0;
    qa('[data-view]').forEach(function (b) { b.classList.toggle("on", +b.dataset.view === k); });
    applyCamera(true);
  }
  function setFit(f) {
    st.fit = f;
    qa('[data-fit]').forEach(function (b) { b.classList.toggle("on", b.dataset.fit === f); });
    applyCamera(true);
  }
  function reset() { st.zoom = 1; st.dh = 0; st.dv = 0; applyCamera(true); }
  function resize() {
    st.W = Math.max(200, +q('[data-el=w]').value || st.W);
    st.H = Math.max(200, +q('[data-el=h]').value || st.H);
    x3d.style.width = st.W + "px"; x3d.style.height = st.H + "px";
    var cv = x3d.querySelector("canvas");          // x3dom 会根据画布实际尺寸自动更新分辨率
    if (cv) { cv.style.width = st.W + "px"; cv.style.height = st.H + "px"; }
    x3d.setAttribute("width", st.W + "px"); x3d.setAttribute("height", st.H + "px");
    applyCamera(true);
  }
  function pan(dh, dv) { st.dh += dh; st.dv += dv; applyCamera(true); }
  function zoomBy(f) { st.zoom = Math.max(0.05, Math.min(50, st.zoom * f)); applyCamera(false); }
  function panStep() { return Math.max(0.3, 2 * geom().base / st.zoom * 0.05); }

  // ---------- 控件事件 ----------
  slider.addEventListener("input", function () { show(+slider.value); });
  q('[data-act=prev]').onclick = function () { show(st.i - 1); };
  q('[data-act=next]').onclick = function () { show(st.i + 1); };
  q('[data-act=play]').onclick = togglePlay;
  q('[data-act=reset]').onclick = reset;
  qa('[data-view]').forEach(function (b) { b.onclick = function () { setView(+b.dataset.view); }; });
  qa('[data-fit]').forEach(function (b) { b.onclick = function () { setFit(b.dataset.fit); }; });
  q('[data-el=w]').addEventListener("change", resize);
  q('[data-el=h]').addEventListener("change", resize);

  // ---------- 鼠标：Ctrl+滚轮缩放，Alt+拖动平移 ----------
  x3d.addEventListener("wheel", function (e) {
    if (!e.ctrlKey) return;
    e.preventDefault(); e.stopPropagation();
    zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, {capture: true, passive: false});

  var drag = null;
  x3d.addEventListener("mousedown", function (e) {
    if (!e.altKey || e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    drag = {x: e.clientX, y: e.clientY};
  }, true);
  window.addEventListener("mousemove", function (e) {
    if (!drag) return;
    e.preventDefault(); e.stopPropagation();
    var s = 2 * geom().base / st.zoom / Math.min(st.W, st.H);
    pan(-(e.clientX - drag.x) * s, (e.clientY - drag.y) * s);
    drag = {x: e.clientX, y: e.clientY};
  }, true);
  window.addEventListener("mouseup", function (e) {
    if (drag) { drag = null; e.preventDefault(); e.stopPropagation(); }
  }, true);

  // ---------- 键盘 ----------
  root.addEventListener("mouseenter", function () {
    var t = document.activeElement;
    if (!t || !root.contains(t) || t === root) root.focus({preventScroll: true});
  });
  root.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" && e.target.type === "number") return;
    var k = e.key, done = true;
    if (k === "ArrowRight" && e.shiftKey) pan(panStep(), 0);
    else if (k === "ArrowLeft" && e.shiftKey) pan(-panStep(), 0);
    else if (k === "ArrowRight") show(st.i + 1);
    else if (k === "ArrowLeft") show(st.i - 1);
    else if (k === "ArrowUp") pan(0, panStep());
    else if (k === "ArrowDown") pan(0, -panStep());
    else if (k === "+" || k === "=") zoomBy(1.2);
    else if (k === "-" || k === "_") zoomBy(1 / 1.2);
    else if (k === "1" || k === "2" || k === "3") setView(+k - 1);
    else if (k === "0") reset();
    else if (k === " ") togglePlay();
    else done = false;
    if (done) { e.preventDefault(); e.stopPropagation(); }
  }, true);

  // ---------- 等 x3dom 初始化完成 ----------
  function init() {
    vp = document.getElementById("vp_" + cfg.uid);
    sw = document.getElementById("sw_" + cfg.uid);
    if (!window.x3dom || !vp || !vp._x3domNode || !sw || !sw._x3domNode) {
      if (window.x3dom && x3dom.reload && !init.reloaded && (init.tries = (init.tries || 0) + 1) > 20) {
        init.reloaded = true; x3dom.reload();       // 重复运行单元格时 x3dom 可能需要手动重新扫描
      }
      setTimeout(init, 50); return;
    }
    F = x3dom.fields;
    show(0);
    applyCamera(true);
  }
  show(0);
  init();
})();
</script>
"""

display(HTML(build_viewer(images, names, CANVAS_W, CANVAS_H, SPHERE_QUALITY, PLAY_INTERVAL_MS)))
#   None       不处理，按文件原样显示
WRAP_MODE = "molecule"


def wrap_into_cell(atoms, mode="molecule", max_mol_size=20):
    """把原子放回晶胞内。
    mode="atom"     : 每个原子单独放回晶胞（最整齐，但跨边界的分子会被拆开）
    mode="molecule" : 小分子整体平移回晶胞（保持分子完整），大的部分（如金属板）逐原子放回
    """
    atoms = atoms.copy()
    if not atoms.pbc.any() or mode is None:
        return atoms
    if mode == "atom":
        atoms.wrap()
        return atoms

    nl = build_neighbor_list(atoms, cutoffs=natural_cutoffs(atoms), self_interaction=False, bothways=True)
    n_comp, labels = connected_components(nl.get_connectivity_matrix(sparse=True), directed=False)

    pos = atoms.positions.copy()
    cell, pbc = atoms.cell, atoms.pbc
    for c in range(n_comp):
        idx = np.where(labels == c)[0]
        if len(idx) > max_mol_size:                 # 金属板等大块：逐原子放回
            frac = cell.scaled_positions(pos[idx])
            frac[:, pbc] %= 1.0
            pos[idx] = cell.cartesian_positions(frac)
            continue
        # 小分子：先用最小镜像把分子拼完整，再把分子中心平移进晶胞
        ref = pos[idx[0]]
        d, _ = find_mic(pos[idx] - ref, cell, pbc)
        mol = ref + d
        center = mol.mean(axis=0)
        fc = cell.scaled_positions(center[None])[0]
        shift_frac = np.where(pbc, np.floor(fc), 0.0)
        pos[idx] = mol - cell.cartesian_positions(shift_frac[None])[0]
    atoms.positions = pos
    return atoms


files = []
for p in patterns:
    files += glob.glob(os.path.join(base_dir, "**", p), recursive=True)
files = sorted(set(files))

images, names = [], []
for f in files:
    try:
        images.append(wrap_into_cell(read(f, index=-1), WRAP_MODE))   # 多帧文件取最后一帧
        names.append(os.path.relpath(f, base_dir))
    except Exception as e:
        print(f"读取失败 {f}: {e}")

print(f"共读取 {len(images)} 个结构")

# =====================================================================
# 2. 查看器设置
# =====================================================================
CANVAS_W, CANVAS_H = 920, 580     # 默认画布大小（页面里也可以改）
SPHERE_QUALITY = "medium"         # 球的精细度：low / medium / high，越低越流畅
PLAY_INTERVAL_MS = 800            # 自动播放时每个结构停留的毫秒数


# =====================================================================
# 3. 生成整页查看器（所有结构一次性放进网页，切换在浏览器里完成）
# =====================================================================
def build_viewer(images, names, width=920, height=580, quality="medium", interval=800):
    uid = uuid.uuid4().hex[:8]

    # 每种元素只定义一次球体，其余原子复用（USE），减少内存和渲染开销
    elements = sorted({int(z) for a in images for z in a.numbers})
    defs = []
    for z in elements:
        r, g, b = jmol_colors[z]
        defs.append(
            f'<shape DEF="el{z}_{uid}"><appearance><material diffuseColor="{r:.3f} {g:.3f} {b:.3f}"></material>'
            f'</appearance><sphere radius="{covalent_radii[z]:.3f}"></sphere></shape>')

    def cell_lines(cell):
        o = np.zeros(3)
        a, b, c = cell
        corners = [o, a, b, c, a + b, a + c, b + c, a + b + c]
        edges = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4), (2, 6),
                 (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
        pts = " ".join(f"{corners[i][0]:.3f} {corners[i][1]:.3f} {corners[i][2]:.3f} "
                       f"{corners[j][0]:.3f} {corners[j][1]:.3f} {corners[j][2]:.3f}"
                       for i, j in edges)
        return ('<shape><appearance><material emissiveColor="0 0 0" diffuseColor="0 0 0"></material></appearance>'
                f'<lineset vertexCount="{" ".join(["2"] * 12)}"><coordinate point="{pts}"></coordinate>'
                '</lineset></shape>')

    groups, meta = [], []
    all_lo, all_hi = np.full(3, np.inf), np.full(3, -np.inf)
    cell_lo, cell_hi = np.full(3, np.inf), np.full(3, -np.inf)
    for atoms, name in zip(images, names):
        parts = []
        if atoms.pbc.any():
            parts.append(cell_lines(atoms.cell))
            c = atoms.cell
            corners = np.array([i * c[0] + j * c[1] + k * c[2]
                                for i in (0, 1) for j in (0, 1) for k in (0, 1)])
            cell_lo, cell_hi = np.minimum(cell_lo, corners.min(0)), np.maximum(cell_hi, corners.max(0))
        for (x, y, zc), z in zip(atoms.positions, atoms.numbers):
            parts.append(f'<transform translation="{x:.3f} {y:.3f} {zc:.3f}">'
                         f'<shape USE="el{int(z)}_{uid}"></shape></transform>')
        groups.append("<group>" + "".join(parts) + "</group>")
        all_lo = np.minimum(all_lo, atoms.positions.min(0))
        all_hi = np.maximum(all_hi, atoms.positions.max(0))
        meta.append(dict(name=name, formula=atoms.get_chemical_formula(), n=len(atoms)))

    if not np.isfinite(cell_lo).all():          # 没有晶胞时，"整个晶胞"模式退化为只看原子
        cell_lo, cell_hi = all_lo, all_hi

    cfg = dict(uid=uid, meta=meta, width=width, height=height, interval=interval,
               atoms_box=[all_lo.tolist(), all_hi.tolist()],
               cell_box=[cell_lo.tolist(), cell_hi.tolist()])

    html = HTML_TEMPLATE
    for key, val in {
        "__UID__": uid,
        "__W__": str(width), "__H__": str(height),
        "__QUALITY__": quality,
        "__DEFS__": "".join(defs),
        "__GROUPS__": "".join(groups),
        "__N__": str(len(images) - 1),
        "__CFG__": json.dumps(cfg),
    }.items():
        html = html.replace(key, val)
    return html


HTML_TEMPLATE = r"""
<link rel="stylesheet" type="text/css" href="https://www.x3dom.org/release/x3dom.css">
<script type="text/javascript" src="https://www.x3dom.org/release/x3dom.js"></script>
<style>
  #av___UID__ { background:#fff; color:#000; font:13px/1.4 sans-serif; padding:10px 12px;
                border-radius:6px; display:inline-block; outline:none; }
  #av___UID__ .row { display:flex; align-items:center; gap:8px; margin:4px 0; flex-wrap:wrap; }
  #av___UID__ button { padding:3px 10px; border:1px solid #bbb; background:#f3f3f3; border-radius:3px;
                       cursor:pointer; color:#000; }
  #av___UID__ button.on { background:#c8c8c8; }
  #av___UID__ select, #av___UID__ input[type=number] { color:#000; background:#fff; border:1px solid #bbb; }
  #av___UID__ input[type=number] { width:70px; }
  #av___UID__ .help { font-size:11px; color:#555; }
</style>

<div id="av___UID__" tabindex="0">
  <div class="row">
    <button data-act="play">▶</button>
    <button data-act="prev">◀</button>
    <input type="range" min="0" max="__N__" value="0" style="width:420px" data-el="slider">
    <button data-act="next">▶|</button>
    <span data-el="idx"></span>
  </div>
  <div class="row">
    视角
    <button data-view="0" class="on">正视 (沿y, z朝上)</button>
    <button data-view="1">侧视 (沿x, z朝上)</button>
    <button data-view="2">俯视 (沿z)</button>
    &nbsp; 框选
    <button data-fit="atoms" class="on">只看原子</button>
    <button data-fit="cell">整个晶胞</button>
  </div>
  <div class="row">
    画布宽 <input type="number" data-el="w" value="__W__" step="10" min="200">
    高 <input type="number" data-el="h" value="__H__" step="10" min="200">
    <button data-act="reset">复位视角 (0)</button>
    <span data-el="zoominfo" class="help"></span>
  </div>
  <div class="help">
    鼠标：Ctrl+滚轮 缩放 | Alt+左键拖动 平移 | 左键拖动 旋转 &nbsp;&nbsp;
    键盘（鼠标放在本区域内）：←/→ 切换结构 | ↑/↓ 上下平移 | Shift+←/→ 左右平移 | +/- 缩放 | 1/2/3 视角 | 0 复位 | 空格 播放
  </div>
  <div class="row"><b data-el="info"></b></div>
  <X3D id="x3d___UID__" width="__W__px" height="__H__px" disableKeys="true"
       style="border:1px solid #ccc; background:#fff;">
    <param name="PrimitiveQuality" value="__QUALITY__">
    <scene>
      <OrthoViewpoint id="vp___UID__" position="0 0 100" fieldOfView="-10 -10 10 10"></OrthoViewpoint>
      <background skyColor="1 1 1"></background>
      <switch whichChoice="-1">__DEFS__</switch>
      <switch id="sw___UID__" whichChoice="0">__GROUPS__</switch>
    </scene>
  </X3D>
</div>

<script>
(function () {
  var cfg = __CFG__;
  var root = document.getElementById("av_" + cfg.uid);
  var x3d = document.getElementById("x3d_" + cfg.uid);
  var q = function (sel) { return root.querySelector(sel); };
  var qa = function (sel) { return root.querySelectorAll(sel); };
  var slider = q('[data-el=slider]');

  var VIEWS = [
    {orient: [1, 0, 0, 1.5708],                 axis: 1, sign: -1, h: 0, v: 2},
    {orient: [0.57735, 0.57735, 0.57735, 2.0944], axis: 0, sign: +1, h: 1, v: 2},
    {orient: [0, 0, 1, 0],                      axis: 2, sign: +1, h: 0, v: 1}
  ];
  var st = {i: 0, view: 0, fit: "atoms", zoom: 1, dh: 0, dv: 0,
            W: cfg.width, H: cfg.height, playing: null};
  var vp, sw, F;

  // ---------- 相机 ----------
  function geom() {
    var box = st.fit === "cell" ? cfg.cell_box : cfg.atoms_box;
    var V = VIEWS[st.view], lo = box[0], hi = box[1];
    var eh = (hi[V.h] - lo[V.h]) * 1.1 + 2, ev = (hi[V.v] - lo[V.v]) * 1.1 + 2;
    var base = st.W <= st.H ? Math.max(eh / 2, ev / 2 * st.W / st.H)
                            : Math.max(ev / 2, eh / 2 * st.H / st.W);
    var c = [0, 1, 2].map(function (k) { return (lo[k] + hi[k]) / 2; });
    return {V: V, base: base, c: c};
  }
  function applyCamera(resetRotation) {
    if (!vp) return;
    var g = geom(), half = g.base / st.zoom;
    vp.setFieldValue("fieldOfView", new F.MFFloat([-half, -half, half, half]));
    if (resetRotation) {
      var c = g.c.slice();
      c[g.V.h] += st.dh; c[g.V.v] += st.dv;
      var p = c.slice();
      p[g.V.axis] += g.V.sign * (g.base * 10 + 50);
      var o = g.V.orient;
      vp.setFieldValue("orientation", F.Quaternion.axisAngle(new F.SFVec3f(o[0], o[1], o[2]), o[3]));
      vp.setFieldValue("centerOfRotation", new F.SFVec3f(c[0], c[1], c[2]));
      vp.setFieldValue("position", new F.SFVec3f(p[0], p[1], p[2]));
    }
    q('[data-el=zoominfo]').textContent = "缩放 " + st.zoom.toFixed(2) + "×";
  }

  // ---------- 结构切换（只改 Switch 节点，不重新加载，非常快） ----------
  function show(i) {
    var n = cfg.meta.length;
    st.i = Math.max(0, Math.min(n - 1, i));
    slider.value = st.i;
    if (sw) sw.setFieldValue("whichChoice", st.i);
    var m = cfg.meta[st.i];
    q('[data-el=idx]').textContent = st.i + " / " + (n - 1);
    q('[data-el=info]').textContent = "[" + st.i + "] " + m.name + "  |  " + m.formula + "  |  " + m.n + " 个原子";
  }
  function togglePlay() {
    var btn = q('[data-act=play]');
    if (st.playing) { clearInterval(st.playing); st.playing = null; btn.textContent = "▶"; return; }
    btn.textContent = "❚❚";
    st.playing = setInterval(function () {
      show(st.i >= cfg.meta.length - 1 ? 0 : st.i + 1);
    }, cfg.interval);
  }
  function setView(k) {
    st.view = k; st.dh = 0; st.dv = 0;
    qa('[data-view]').forEach(function (b) { b.classList.toggle("on", +b.dataset.view === k); });
    applyCamera(true);
  }
  function setFit(f) {
    st.fit = f;
    qa('[data-fit]').forEach(function (b) { b.classList.toggle("on", b.dataset.fit === f); });
    applyCamera(true);
  }
  function reset() { st.zoom = 1; st.dh = 0; st.dv = 0; applyCamera(true); }
  function resize() {
    st.W = Math.max(200, +q('[data-el=w]').value || st.W);
    st.H = Math.max(200, +q('[data-el=h]').value || st.H);
    x3d.style.width = st.W + "px"; x3d.style.height = st.H + "px";
    var cv = x3d.querySelector("canvas");          // x3dom 会根据画布实际尺寸自动更新分辨率
    if (cv) { cv.style.width = st.W + "px"; cv.style.height = st.H + "px"; }
    x3d.setAttribute("width", st.W + "px"); x3d.setAttribute("height", st.H + "px");
    applyCamera(true);
  }
  function pan(dh, dv) { st.dh += dh; st.dv += dv; applyCamera(true); }
  function zoomBy(f) { st.zoom = Math.max(0.05, Math.min(50, st.zoom * f)); applyCamera(false); }
  function panStep() { return Math.max(0.3, 2 * geom().base / st.zoom * 0.05); }

  // ---------- 控件事件 ----------
  slider.addEventListener("input", function () { show(+slider.value); });
  q('[data-act=prev]').onclick = function () { show(st.i - 1); };
  q('[data-act=next]').onclick = function () { show(st.i + 1); };
  q('[data-act=play]').onclick = togglePlay;
  q('[data-act=reset]').onclick = reset;
  qa('[data-view]').forEach(function (b) { b.onclick = function () { setView(+b.dataset.view); }; });
  qa('[data-fit]').forEach(function (b) { b.onclick = function () { setFit(b.dataset.fit); }; });
  q('[data-el=w]').addEventListener("change", resize);
  q('[data-el=h]').addEventListener("change", resize);

  // ---------- 鼠标：Ctrl+滚轮缩放，Alt+拖动平移 ----------
  x3d.addEventListener("wheel", function (e) {
    if (!e.ctrlKey) return;
    e.preventDefault(); e.stopPropagation();
    zoomBy(e.deltaY < 0 ? 1.1 : 1 / 1.1);
  }, {capture: true, passive: false});

  var drag = null;
  x3d.addEventListener("mousedown", function (e) {
    if (!e.altKey || e.button !== 0) return;
    e.preventDefault(); e.stopPropagation();
    drag = {x: e.clientX, y: e.clientY};
  }, true);
  window.addEventListener("mousemove", function (e) {
    if (!drag) return;
    e.preventDefault(); e.stopPropagation();
    var s = 2 * geom().base / st.zoom / Math.min(st.W, st.H);
    pan(-(e.clientX - drag.x) * s, (e.clientY - drag.y) * s);
    drag = {x: e.clientX, y: e.clientY};
  }, true);
  window.addEventListener("mouseup", function (e) {
    if (drag) { drag = null; e.preventDefault(); e.stopPropagation(); }
  }, true);

  // ---------- 键盘 ----------
  root.addEventListener("mouseenter", function () {
    var t = document.activeElement;
    if (!t || !root.contains(t) || t === root) root.focus({preventScroll: true});
  });
  root.addEventListener("keydown", function (e) {
    if (e.target.tagName === "INPUT" && e.target.type === "number") return;
    var k = e.key, done = true;
    if (k === "ArrowRight" && e.shiftKey) pan(panStep(), 0);
    else if (k === "ArrowLeft" && e.shiftKey) pan(-panStep(), 0);
    else if (k === "ArrowRight") show(st.i + 1);
    else if (k === "ArrowLeft") show(st.i - 1);
    else if (k === "ArrowUp") pan(0, panStep());
    else if (k === "ArrowDown") pan(0, -panStep());
    else if (k === "+" || k === "=") zoomBy(1.2);
    else if (k === "-" || k === "_") zoomBy(1 / 1.2);
    else if (k === "1" || k === "2" || k === "3") setView(+k - 1);
    else if (k === "0") reset();
    else if (k === " ") togglePlay();
    else done = false;
    if (done) { e.preventDefault(); e.stopPropagation(); }
  }, true);

  // ---------- 等 x3dom 初始化完成 ----------
  function init() {
    vp = document.getElementById("vp_" + cfg.uid);
    sw = document.getElementById("sw_" + cfg.uid);
    if (!window.x3dom || !vp || !vp._x3domNode || !sw || !sw._x3domNode) {
      if (window.x3dom && x3dom.reload && !init.reloaded && (init.tries = (init.tries || 0) + 1) > 20) {
        init.reloaded = true; x3dom.reload();       // 重复运行单元格时 x3dom 可能需要手动重新扫描
      }
      setTimeout(init, 50); return;
    }
    F = x3dom.fields;
    show(0);
    applyCamera(true);
  }
  show(0);
  init();
})();
</script>
"""

display(HTML(build_viewer(images, names, CANVAS_W, CANVAS_H, SPHERE_QUALITY, PLAY_INTERVAL_MS)))
