# -*- coding: utf-8 -*-
"""
Demo web app: Tối ưu khung thép 2D bằng OpenSeesPy (phân tích phi tuyến nâng cao)
+ NSGA-II (pymoo).

- Nhập nút, phần tử, liên kết, tải trọng -> hình khung cập nhật tức thì.
- Tất cả CỘT dùng chung 1 tiết diện I, tất cả DẦM dùng chung 1 tiết diện I.
- Biến thiết kế: h, tw, bf, tf của cột và dầm + Fy (mác thép).
- Mục tiêu: (1) khối lượng thép nhỏ nhất, (2) Fy nhỏ nhất.
- Ràng buộc: hệ số tải giới hạn lambda >= lambda mục tiêu, tiết diện không mảnh.

Chạy:  streamlit run app.py
"""
import math
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import openseespy.opensees as ops
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize

# OpenSees dùng trạng thái toàn cục -> khóa để nhiều người dùng không chạy chồng lên nhau
OPS_LOCK = threading.Lock()

# Đơn vị nội bộ: m, N, Pa
mm, kN, MPa, GPa = 1e-3, 1e3, 1e6, 1e9
FIX_TYPES = {
    "Tự do": (0, 0, 0),
    "Ngàm": (1, 1, 1),
    "Khớp": (1, 1, 0),
    "Con lăn (giữ Y)": (0, 1, 0),
}
VAR_NAMES = ["h_col", "tw_col", "bf_col", "tf_col",
             "h_beam", "tw_beam", "bf_beam", "tf_beam"]

st.set_page_config(page_title="Tối ưu khung thép 2D – NSGA-II", layout="wide")


# =====================================================================
# 1. DỮ LIỆU MẶC ĐỊNH (khung cổng trong notebook)
# =====================================================================
def default_data():
    nodes = pd.DataFrame({
        "Nút": [1, 2, 3, 4],
        "X (m)": [0.0, 0.0, 4.0, 4.0],
        "Y (m)": [0.0, 5.0, 5.0, 0.0],
        "Liên kết": ["Ngàm", "Tự do", "Tự do", "Ngàm"],
    })
    elems = pd.DataFrame({
        "Phần tử": [1, 2, 3],
        "Nút i": [1, 4, 2],
        "Nút j": [2, 3, 3],
        "Loại": ["Cột", "Cột", "Dầm"],
    })
    loads = pd.DataFrame({
        "Nút": [2, 3],
        "Fx (kN)": [35.0, 0.0],
        "Fy (kN)": [-800.0, -800.0],
        "M (kNm)": [0.0, 0.0],
    })
    return nodes, elems, loads


if "nodes" not in st.session_state:
    n, e, l = default_data()
    st.session_state.nodes, st.session_state.elems, st.session_state.loads = n, e, l


# =====================================================================
# 2. KIỂM TRA & CHUẨN HÓA DỮ LIỆU NHẬP
# =====================================================================
def clean_model(nodes_df, elems_df, loads_df):
    """Trả về (model dict, danh sách lỗi, danh sách cảnh báo)."""
    errs, warns = [], []
    nodes = {}
    for _, r in nodes_df.dropna(subset=["Nút", "X (m)", "Y (m)"]).iterrows():
        nid = int(r["Nút"])
        if nid in nodes:
            errs.append(f"Nút {nid} bị trùng số hiệu.")
        fix = r["Liên kết"] if r["Liên kết"] in FIX_TYPES else "Tự do"
        nodes[nid] = (float(r["X (m)"]), float(r["Y (m)"]), fix)

    elems = {}
    for _, r in elems_df.dropna(subset=["Phần tử", "Nút i", "Nút j"]).iterrows():
        eid, i, j = int(r["Phần tử"]), int(r["Nút i"]), int(r["Nút j"])
        typ = r["Loại"] if r["Loại"] in ("Cột", "Dầm") else None
        if eid in elems:
            errs.append(f"Phần tử {eid} bị trùng số hiệu.")
        if i not in nodes or j not in nodes:
            errs.append(f"Phần tử {eid}: nút {i} hoặc {j} chưa được khai báo.")
            continue
        if i == j:
            errs.append(f"Phần tử {eid}: nút đầu trùng nút cuối.")
            continue
        (xi, yi, _), (xj, yj, _) = nodes[i], nodes[j]
        L = math.hypot(xj - xi, yj - yi)
        if L < 1e-9:
            errs.append(f"Phần tử {eid} có chiều dài bằng 0.")
            continue
        if typ is None:  # tự phân loại theo phương
            typ = "Cột" if abs(yj - yi) > abs(xj - xi) else "Dầm"
        elems[eid] = (i, j, typ, L)

    loads = []
    for _, r in loads_df.dropna(subset=["Nút"]).iterrows():
        nid = int(r["Nút"])
        if nid not in nodes:
            errs.append(f"Tải trọng tại nút {nid}: nút chưa được khai báo.")
            continue
        num = lambda v: 0.0 if pd.isna(v) else float(v)
        loads.append((nid, num(r.get("Fx (kN)", 0)), num(r.get("Fy (kN)", 0)),
                      num(r.get("M (kNm)", 0))))

    if not any(v[2] != "Tự do" for v in nodes.values()):
        errs.append("Chưa có nút nào có liên kết (ngàm/khớp) – kết cấu biến hình.")
    if not elems:
        errs.append("Chưa có phần tử nào.")
    if not loads:
        warns.append("Chưa có tải trọng nút (chỉ còn tải bản thân).")
    if not any(v[2] == "Cột" for v in elems.values()):
        warns.append("Không có phần tử nào là Cột.")
    if not any(v[2] == "Dầm" for v in elems.values()):
        warns.append("Không có phần tử nào là Dầm.")
    used = {v[0] for v in elems.values()} | {v[1] for v in elems.values()}
    lonely = sorted(set(nodes) - used)
    if lonely:
        warns.append(f"Nút không nối với phần tử nào: {lonely} (bỏ qua khi phân tích).")
    return {"nodes": nodes, "elems": elems, "loads": loads}, errs, warns


# =====================================================================
# 3. HÌNH VẼ KHUNG
# =====================================================================
def draw_frame(model, disp=None, scale=1.0):
    fig = go.Figure()
    nodes, elems = model["nodes"], model["elems"]
    color = {"Cột": "#1f6feb", "Dầm": "#d9480f"}
    shown = set()
    for eid, (i, j, typ, _) in elems.items():
        xi, yi, _ = nodes[i]; xj, yj, _ = nodes[j]
        fig.add_trace(go.Scatter(
            x=[xi, xj], y=[yi, yj], mode="lines",
            line=dict(color=color[typ], width=5),
            name=typ, legendgroup=typ, showlegend=typ not in shown,
            hovertemplate=f"PT {eid}: {i}→{j} ({typ})<extra></extra>"))
        shown.add(typ)
        fig.add_annotation(x=(xi + xj) / 2, y=(yi + yj) / 2, text=f"[{eid}]",
                           showarrow=False, font=dict(size=11, color=color[typ]),
                           bgcolor="rgba(255,255,255,0.8)")
        if disp is not None:
            ui, uj = disp.get(i, (0, 0)), disp.get(j, (0, 0))
            fig.add_trace(go.Scatter(
                x=[xi + scale * ui[0], xj + scale * uj[0]],
                y=[yi + scale * ui[1], yj + scale * uj[1]],
                mode="lines", line=dict(color="gray", width=2, dash="dash"),
                name="Biến dạng", legendgroup="def",
                showlegend="def" not in shown, hoverinfo="skip"))
            shown.add("def")

    xs = [v[0] for v in nodes.values()]; ys = [v[1] for v in nodes.values()]
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers+text", text=[str(k) for k in nodes],
        textposition="top left", marker=dict(size=9, color="black"),
        name="Nút", hovertemplate="Nút %{text}: (%{x}, %{y})<extra></extra>"))

    span = max(max(xs) - min(xs) if xs else 1, max(ys) - min(ys) if ys else 1, 1.0)
    s = 0.04 * span
    sym = {"Ngàm": "square", "Khớp": "triangle-up", "Con lăn (giữ Y)": "circle-open"}
    for nid, (x, y, fix) in nodes.items():
        if fix != "Tự do":
            fig.add_trace(go.Scatter(
                x=[x], y=[y - s * 0.6], mode="markers",
                marker=dict(symbol=sym[fix], size=16, color="#2f9e44"),
                showlegend=False, hovertemplate=f"{fix}<extra></extra>"))

    for nid, fx, fy, mz in model["loads"]:
        x, y, _ = nodes[nid]
        if abs(fx) > 0:
            fig.add_annotation(x=x, y=y, ax=-60 * np.sign(fx), ay=0,
                               arrowhead=2, arrowwidth=2, arrowcolor="#c92a2a",
                               text=f"{fx:g} kN", font=dict(color="#c92a2a"))
        if abs(fy) > 0:
            fig.add_annotation(x=x, y=y, ax=0, ay=60 * np.sign(fy),
                               arrowhead=2, arrowwidth=2, arrowcolor="#c92a2a",
                               text=f"{fy:g} kN", font=dict(color="#c92a2a"))
        if abs(mz) > 0:
            fig.add_annotation(x=x, y=y, text=f"↺ {mz:g} kNm", showarrow=False,
                               yshift=-18, font=dict(color="#c92a2a"))

    pad = 0.15 * span
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="X (m)", range=[min(xs or [0]) - pad, max(xs or [1]) + pad],
                   zeroline=False),
        yaxis=dict(title="Y (m)", scaleanchor="x", scaleratio=1, zeroline=False,
                   range=[min(ys or [0]) - pad, max(ys or [1]) + pad]),
        legend=dict(orientation="h", y=1.06), plot_bgcolor="white")
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# =====================================================================
# 4. PHÂN TÍCH OPENSEES (tổng quát hóa từ cal_pu trong notebook)
# =====================================================================
def section_area(h, tw, bf, tf):  # mm -> m2
    return (2 * bf * tf + (h - 2 * tf) * tw) * mm ** 2


def build_and_analyze(model, sec_col, sec_beam, Fy_MPa, p, record=False):
    """Phân tích tĩnh phi tuyến hình học + vật liệu, điều khiển tải.
    Trả về lambda_max (và đường cong nếu record=True)."""
    E, b = p["E_GPa"] * GPa, p["b"]
    Fy = Fy_MPa * MPa
    elems = model["elems"]
    used = {v[0] for v in elems.values()} | {v[1] for v in elems.values()}
    nodes = {k: v for k, v in model["nodes"].items() if k in used}

    ops.wipe()
    ops.model("basic", "-ndm", 2, "-ndf", 3)
    ops.uniaxialMaterial("Steel02", 1, Fy, E, b, 20, 0.925, 0.15)

    def fiber_section(secTag, h, bf, tw, tf):
        # Ứng suất dư theo ECCS (như notebook)
        ops.section("Fiber", secTag)
        sig_r = (0.5 if h / bf <= 1.2 else 0.3) * Fy * p["use_residual"]
        matID = secTag * 5000
        n_f, n_w = 20, 30
        for y_f in (h / 2 - tf / 2, -h / 2 + tf / 2):
            for i in range(n_f):
                x_f = -bf / 2 + (i + 0.5) * (bf / n_f)
                res = sig_r * (1 - 2 * abs(2 * x_f / bf))
                matID += 1
                ops.uniaxialMaterial("InitStrainMaterial", matID, 1, res / E)
                ops.fiber(y_f, x_f, tf * (bf / n_f), matID)
        h_web = h - 2 * tf
        for i in range(n_w):
            y_w = -h_web / 2 + (i + 0.5) * (h_web / n_w)
            res_w = sig_r * (2 * abs(2 * y_w / h_web) - 1)
            matID += 1
            ops.uniaxialMaterial("InitStrainMaterial", matID, 1, res_w / E)
            ops.fiber(y_w, 0.0, (h_web / n_w) * tw, matID)

    hc, twc, bfc, tfc = sec_col
    hb, twb, bfb, tfb = sec_beam
    fiber_section(1, hc * mm, bfc * mm, twc * mm, tfc * mm)
    fiber_section(2, hb * mm, bfb * mm, twb * mm, tfb * mm)
    A = {"Cột": section_area(*sec_col), "Dầm": section_area(*sec_beam)}

    # Nút + sai lệch hình học ban đầu: dịch ngang tỉ lệ với cao độ (H/ratio)
    ys = [v[1] for v in nodes.values()]
    y0, H = min(ys), max(ys) - min(ys)
    delta = (H / p["imp_ratio"]) if (H > 0 and p["imp_ratio"] > 0) else 0.0
    for nid, (x, y, fix) in nodes.items():
        dx = delta * (y - y0) / H if H > 0 else 0.0
        ops.node(nid, x + dx, y)
        if fix != "Tự do":
            ops.fix(nid, *FIX_TYPES[fix])

    ops.geomTransf("PDelta", 1)
    ops.beamIntegration("Lobatto", 1, 1, p["n_ip"])
    ops.beamIntegration("Lobatto", 2, 2, p["n_ip"])
    for eid, (i, j, typ, _) in elems.items():
        ops.element("forceBeamColumn", eid, i, j, 1, 1 if typ == "Cột" else 2)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for nid, fx, fy, mz in model["loads"]:
        if nid in nodes:
            ops.load(nid, fx * kN, fy * kN, mz * kN)
    if p["self_weight"]:
        gamma = p["gamma_kNm3"] * kN  # N/m3
        for eid, (i, j, typ, L) in elems.items():
            xi, yi, _ = nodes[i]; xj, yj, _ = nodes[j]
            c, s = (xj - xi) / L, (yj - yi) / L
            w = A[typ] * gamma                # N/m, hướng -Y toàn cục
            ops.eleLoad("-ele", eid, "-type", "-beamUniform", -w * c, -w * s)

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-7, 100)
    ops.algorithm("Newton")
    ops.integrator("LoadControl", p["dlam"])
    ops.analysis("Static")

    ctrl = p["ctrl_node"] if p["ctrl_node"] in nodes else max(nodes, key=lambda k: nodes[k][1])
    lam_max, dlam, n_cut = 0.0, p["dlam"], 0
    curve = [(0.0, 0.0)]
    for _ in range(int(p["lam_cap"] / p["dlam"]) * 4):
        if ops.analyze(1) != 0:
            if n_cut >= p["max_cut"]:
                break
            n_cut += 1
            dlam /= 2.0
            ops.integrator("LoadControl", dlam)
            continue
        lam_max = ops.getTime()
        if record:
            curve.append((ops.nodeDisp(ctrl, 1) * 1000, lam_max))
        if lam_max >= p["lam_cap"] - 1e-9:
            break
    disp = None
    if record:
        disp = {nid: (ops.nodeDisp(nid, 1), ops.nodeDisp(nid, 2)) for nid in nodes}
    ops.wipe()
    return (lam_max, curve, disp) if record else lam_max


def total_weight(model, sec_col, sec_beam, rho=7850.0):
    A = {"Cột": section_area(*sec_col), "Dầm": section_area(*sec_beam)}
    return sum(A[t] * L * rho for (_, _, t, L) in model["elems"].values())


# Kiểm tra độ mảnh (giữ nguyên logic notebook, AISC 360)
def check_slender(kind, d, tw, bf, tf, Fy, E):
    h = d - 2 * tf
    bftf, htw = bf / (2 * tf), h / tw
    EFy = math.sqrt(E / Fy)
    if bftf < 0.38 * EFy:
        section = 0
    elif bftf < EFy:
        section = 1
    else:
        section = 2
    kc = min(max(4 / math.sqrt(htw), 0.35), 0.76)
    if kind == "column":
        ok = bftf <= 0.64 * math.sqrt(kc * E / Fy) and htw <= 1.49 * EFy
    elif section < 2:
        ok = bftf <= 0.38 * EFy and htw <= 3.76 * EFy
    else:
        ok = bftf <= 0.95 * math.sqrt(kc * E / (0.7 * Fy)) and htw <= 5.7 * EFy
    return -1 if ok else 1


# =====================================================================
# 5. BÀI TOÁN NSGA-II
# =====================================================================
class FrameProblem(ElementwiseProblem):
    def __init__(self, model, p, xl, xu, fy_grades):
        self.model, self.p, self.fy_grades = model, p, fy_grades
        self.cache = {}
        super().__init__(n_var=9, n_obj=2, n_ieq_constr=3,
                         xl=np.array(xl, float), xu=np.array(xu, float), vtype=int)

    def decode(self, x):
        x = [int(round(v)) for v in x]
        Fy = self.fy_grades[x[8]] if self.fy_grades else x[8]
        return tuple(x[:4]), tuple(x[4:8]), Fy

    def _evaluate(self, x, out, *args, **kwargs):
        sc, sb, Fy = self.decode(x)
        key = (sc, sb, Fy)
        if key not in self.cache:
            E = self.p["E_GPa"] * 1000  # MPa
            g2 = check_slender("column", *sc, Fy, E)
            g3 = check_slender("beam", *sb, Fy, E)
            bad = any(s[0] - 2 * s[3] <= 0 for s in (sc, sb))
            if bad:
                g2 = g3 = 1
            if bad or (self.p["skip_slender"] and (g2 > 0 or g3 > 0)):
                lam = 0.0  # bỏ qua phân tích cho tiết diện mảnh -> nhanh hơn
            else:
                with OPS_LOCK:
                    lam = build_and_analyze(self.model, sc, sb, Fy, self.p)
            W = total_weight(self.model, sc, sb, self.p["rho"])
            self.cache[key] = ([W, Fy], [self.p["lam_target"] - lam, g2, g3], lam)
        F, G, _ = self.cache[key]
        out["F"], out["G"] = F, G


class StProgress(Callback):
    def __init__(self, n_gen, bar, txt):
        super().__init__()
        self.n_gen, self.bar, self.txt = n_gen, bar, txt

    def notify(self, algorithm):
        g = algorithm.n_gen
        self.bar.progress(min(g / self.n_gen, 1.0))
        opt = algorithm.opt
        nf = 0 if opt is None else int(np.sum(np.all(opt.get("G") <= 0, axis=1)))
        self.txt.write(f"Thế hệ {g}/{self.n_gen} · số lần phân tích: "
                       f"{algorithm.evaluator.n_eval} · nghiệm khả thi trên front: {nf}")


# =====================================================================
# 6. GIAO DIỆN
# =====================================================================
st.title("Tối ưu khung thép 2D · OpenSeesPy + NSGA-II")
st.caption("Phân tích phi tuyến nâng cao (P-Δ, ứng suất dư ECCS, sai lệch ban đầu) – "
           "tất cả cột chung 1 tiết diện I, tất cả dầm chung 1 tiết diện I.")

tab_geo, tab_mat, tab_run = st.tabs(["① Hình học & tải", "② Vật liệu & thông số", "③ Chạy tối ưu"])

with tab_geo:
    left, right = st.columns([1, 1.15])
    with left:
        if st.button("↺ Khôi phục khung mẫu (khung cổng 4 m × 5 m)"):
            n, e, l = default_data()
            st.session_state.nodes, st.session_state.elems, st.session_state.loads = n, e, l
            for k in ("ed_nodes", "ed_elems", "ed_loads"):
                st.session_state.pop(k, None)
            st.rerun()
        st.markdown("**Nút**")
        nodes_df = st.data_editor(
            st.session_state.nodes, key="ed_nodes", num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={
                "Nút": st.column_config.NumberColumn(step=1, format="%d", required=True),
                "X (m)": st.column_config.NumberColumn(format="%.3f", required=True),
                "Y (m)": st.column_config.NumberColumn(format="%.3f", required=True),
                "Liên kết": st.column_config.SelectboxColumn(
                    options=list(FIX_TYPES), default="Tự do", required=True),
            })
        st.markdown("**Phần tử** (nút i → nút j)")
        elems_df = st.data_editor(
            st.session_state.elems, key="ed_elems", num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={
                "Phần tử": st.column_config.NumberColumn(step=1, format="%d", required=True),
                "Nút i": st.column_config.NumberColumn(step=1, format="%d", required=True),
                "Nút j": st.column_config.NumberColumn(step=1, format="%d", required=True),
                "Loại": st.column_config.SelectboxColumn(
                    options=["Cột", "Dầm"], help="Để trống: tự nhận theo phương"),
            })
        st.markdown("**Tải trọng nút** (Fy âm = hướng xuống)")
        loads_df = st.data_editor(
            st.session_state.loads, key="ed_loads", num_rows="dynamic",
            use_container_width=True, hide_index=True,
            column_config={"Nút": st.column_config.NumberColumn(step=1, format="%d")})

    model, errs, warns = clean_model(nodes_df, elems_df, loads_df)
    with right:
        st.markdown("**Sơ đồ khung**")
        if model["nodes"]:
            st.plotly_chart(draw_frame(model), use_container_width=True)
        for e_ in errs:
            st.error(e_)
        for w_ in warns:
            st.warning(w_)
        if not errs:
            nc = sum(1 for v in model["elems"].values() if v[2] == "Cột")
            nb = len(model["elems"]) - nc
            st.success(f"Mô hình hợp lệ: {len(model['nodes'])} nút, {nc} cột, {nb} dầm.")

with tab_mat:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Vật liệu")
        E_GPa = st.number_input("E (GPa)", value=205.0, min_value=1.0)
        b_hard = st.number_input("Hệ số tái bền b (Steel02)", value=0.045, format="%.3f")
        rho = st.number_input("Khối lượng riêng ρ (kg/m³) – tính khối lượng", value=7850.0)
        gamma = st.number_input("Trọng lượng riêng γ (kN/m³) – tải bản thân", value=78.5)
        use_res = st.checkbox("Kể ứng suất dư ECCS", value=True)
        self_w = st.checkbox("Kể tải bản thân (phân bố trên phần tử)", value=True)
        st.subheader("Mác thép Fy")
        discrete = st.checkbox("Dùng danh sách mác thép rời rạc", value=False)
        if discrete:
            grades_txt = st.text_input("Fy (MPa), cách nhau dấu phẩy", "235, 275, 355, 420, 460, 550")
            try:
                fy_grades = sorted({float(s) for s in grades_txt.split(",") if s.strip()})
            except ValueError:
                fy_grades = [235.0, 355.0]
                st.error("Danh sách Fy không hợp lệ.")
            fy_lo, fy_hi = 0, len(fy_grades) - 1
        else:
            fy_grades = None
            fy_lo, fy_hi = st.slider("Khoảng Fy (MPa)", 200, 700, (235, 550))
    with c2:
        st.subheader("Giới hạn tiết diện (mm)")
        st.caption("Cột")
        bounds = {}
        defaults = {"h": (80, 1100), "tw": (3, 30), "bf": (50, 450), "tf": (3, 30)}
        for grp, lab in (("col", "cột"), ("beam", "dầm")):
            if grp == "beam":
                st.caption("Dầm")
            cc = st.columns(4)
            for k, (key, (lo, hi)) in enumerate(defaults.items()):
                with cc[k]:
                    a = st.number_input(f"{key} min", value=lo, min_value=1, key=f"{grp}{key}lo")
                    b_ = st.number_input(f"{key} max", value=hi, min_value=1, key=f"{grp}{key}hi")
                    bounds[f"{key}_{grp}"] = (int(a), int(max(a, b_)))
    with c3:
        st.subheader("Phân tích")
        lam_target = st.number_input("Hệ số tải yêu cầu λ ≥", value=1.0, step=0.1)
        imp_ratio = st.number_input("Sai lệch ban đầu: H /", value=450.0, min_value=0.0,
                                    help="0 = không kể sai lệch")
        dlam = st.number_input("Bước tải Δλ", value=0.01, format="%.3f", min_value=0.001)
        n_ip = st.slider("Số điểm tích phân Lobatto", 3, 7, 5)
        stop_at_target = st.checkbox("Dừng khi đạt λ yêu cầu (nhanh hơn nhiều)", value=True)
        skip_sl = st.checkbox("Bỏ qua phân tích nếu tiết diện mảnh", value=True)
        node_ids = sorted(model["nodes"]) or [1]
        ctrl_node = st.selectbox("Nút theo dõi chuyển vị", node_ids,
                                 index=node_ids.index(2) if 2 in node_ids else 0)
        st.subheader("NSGA-II")
        pop = st.number_input("Kích thước quần thể", value=40, min_value=8, step=4)
        noff = st.number_input("Số con mỗi thế hệ", value=20, min_value=4, step=2)
        ngen = st.number_input("Số thế hệ", value=30, min_value=1)
        seed = st.number_input("Seed", value=10, step=1)

params = dict(E_GPa=E_GPa, b=b_hard, gamma_kNm3=gamma, rho=rho, use_residual=1.0 if use_res else 0.0,
              self_weight=self_w, imp_ratio=imp_ratio, dlam=dlam, n_ip=n_ip,
              lam_target=lam_target, lam_cap=lam_target if stop_at_target else 10.0,
              max_cut=4, skip_slender=skip_sl, ctrl_node=ctrl_node)

with tab_run:
    est = int(pop + noff * (ngen - 1))
    st.write(f"Ước tính tối đa **{est}** lần phân tích OpenSees "
             f"(trùng nghiệm được dùng lại từ bộ nhớ đệm).")
    run = st.button("▶ Chạy tối ưu", type="primary", disabled=bool(errs))
    if run:
        xl = [bounds[f"{k}_col"][0] for k in defaults] + [bounds[f"{k}_beam"][0] for k in defaults] + [fy_lo]
        xu = [bounds[f"{k}_col"][1] for k in defaults] + [bounds[f"{k}_beam"][1] for k in defaults] + [fy_hi]
        problem = FrameProblem(model, params, xl, xu, fy_grades)
        algo = NSGA2(pop_size=int(pop), n_offsprings=int(noff),
                     sampling=IntegerRandomSampling(),
                     crossover=SBX(prob=0.9, eta=15, vtype=float, repair=RoundingRepair()),
                     mutation=PM(eta=20, vtype=float, repair=RoundingRepair()),
                     eliminate_duplicates=True)
        bar, txt = st.progress(0.0), st.empty()
        with st.spinner("Đang tối ưu..."):
            res = minimize(problem, algo, ("n_gen", int(ngen)), seed=int(seed),
                           callback=StProgress(int(ngen), bar, txt), verbose=False)
        if res.X is None:
            st.session_state.result = None
            st.error("Không tìm được nghiệm khả thi. Hãy nới giới hạn tiết diện, "
                     "tăng số thế hệ hoặc giảm λ yêu cầu.")
        else:
            X = np.atleast_2d(res.X)
            rows = []
            for x in X:
                sc, sb, Fy = problem.decode(x)
                F, G, lam = problem.cache[(sc, sb, Fy)]
                rows.append(dict(zip(VAR_NAMES, sc + sb), Fy_MPa=Fy,
                                 KhoiLuong_kg=round(F[0], 1), lambda_=round(lam, 3)))
            df = (pd.DataFrame(rows).drop_duplicates()
                  .sort_values("KhoiLuong_kg").reset_index(drop=True))
            df.index = df.index + 1
            st.session_state.result = dict(df=df, model=model, params=params)

    R = st.session_state.get("result")
    if R:
        df = R["df"]
        st.subheader(f"Tập nghiệm Pareto ({len(df)} nghiệm)")
        cA, cB = st.columns([1.1, 1])
        with cA:
            fig = go.Figure(go.Scatter(
                x=df["KhoiLuong_kg"], y=df["Fy_MPa"], mode="markers+lines",
                line=dict(dash="dot", color="#adb5bd"),
                marker=dict(size=11, color="#d9480f", line=dict(width=1, color="black")),
                text=[f"#{i}" for i in df.index],
                customdata=df[VAR_NAMES].values,
                hovertemplate="%{text}<br>Khối lượng: %{x:.1f} kg<br>Fy: %{y} MPa<br>"
                              "Cột: %{customdata[0]}×%{customdata[1]}×%{customdata[2]}×%{customdata[3]}<br>"
                              "Dầm: %{customdata[4]}×%{customdata[5]}×%{customdata[6]}×%{customdata[7]}"
                              "<extra></extra>"))
            fig.update_layout(height=430, xaxis_title="Khối lượng thép (kg)",
                              yaxis_title="Fy (MPa)", margin=dict(t=20, l=10, r=10, b=10),
                              plot_bgcolor="white")
            fig.update_xaxes(showgrid=True, gridcolor="#eee")
            fig.update_yaxes(showgrid=True, gridcolor="#eee")
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            st.dataframe(df.rename(columns={"lambda_": "λ"}), use_container_width=True, height=430)
            st.download_button("⬇ Tải CSV", df.to_csv().encode("utf-8-sig"),
                               "pareto_solutions.csv", "text/csv")

        st.subheader("Kiểm tra một nghiệm")
        pick = st.selectbox("Chọn nghiệm", df.index,
                            format_func=lambda i: f"#{i} – {df.loc[i, 'KhoiLuong_kg']} kg, "
                                                  f"Fy {df.loc[i, 'Fy_MPa']} MPa")
        if st.button("Chạy phân tích đầy đủ cho nghiệm này"):
            r = df.loc[pick]
            sc = tuple(int(r[k]) for k in VAR_NAMES[:4])
            sb = tuple(int(r[k]) for k in VAR_NAMES[4:])
            p2 = dict(R["params"], lam_cap=10.0)  # chạy đến khi mất ổn định
            with OPS_LOCK, st.spinner("Đang phân tích..."):
                lam, curve, disp = build_and_analyze(R["model"], sc, sb, float(r["Fy_MPa"]),
                                                     p2, record=True)
            c1, c2 = st.columns(2)
            with c1:
                d, l = zip(*curve)
                f2 = go.Figure(go.Scatter(x=d, y=l, mode="lines", line=dict(width=3, color="#2f9e44")))
                f2.add_hline(y=R["params"]["lam_target"], line_dash="dash", line_color="red")
                f2.update_layout(height=380, title=f"λ_max = {lam:.3f}",
                                 xaxis_title=f"Chuyển vị ngang nút {R['params']['ctrl_node']} (mm)",
                                 yaxis_title="λ", plot_bgcolor="white")
                st.plotly_chart(f2, use_container_width=True)
            with c2:
                umax = max(max(abs(u[0]), abs(u[1])) for u in disp.values()) or 1e-9
                span = max(v[0] for v in R["model"]["nodes"].values()) - \
                       min(v[0] for v in R["model"]["nodes"].values()) or 1.0
                sc_f = 0.1 * span / umax
                f3 = draw_frame(R["model"], disp, sc_f)
                f3.update_layout(height=380, title=f"Biến dạng tại λ_max (phóng đại ×{sc_f:.0f})")
                st.plotly_chart(f3, use_container_width=True)
