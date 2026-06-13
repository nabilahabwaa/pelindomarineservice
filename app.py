import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_hex
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import io
import warnings
warnings.filterwarnings('ignore')

# ── KONFIGURASI HALAMAN ──────────────────────────────────────
st.set_page_config(
    page_title="K-Means Clustering — PT Pelindo Marine Service",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── DATABASE AKUN (session state) ────────────────────────────
if 'accounts' not in st.session_state:
    st.session_state['accounts'] = {
        'keuanganpms@gmail.com': 'keuangan123'
    }
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'current_user' not in st.session_state:
    st.session_state['current_user'] = ''
if 'page' not in st.session_state:
    st.session_state['page'] = 'login'
if 'manual_df' not in st.session_state:
    st.session_state['manual_df'] = None
if 'use_manual' not in st.session_state:
    st.session_state['use_manual'] = False

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    .login-container {
        max-width: 420px;
        margin: 60px auto;
        background: white;
        border-radius: 16px;
        padding: 40px 36px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    }
    .login-title {
        font-size: 22px;
        font-weight: 700;
        color: #0a3d62;
        margin-bottom: 4px;
    }
    .login-sub {
        font-size: 13px;
        color: #7f8c8d;
        margin-bottom: 24px;
    }
    .link-btn {
        background: none;
        border: none;
        color: #2980b9;
        cursor: pointer;
        font-size: 13px;
        padding: 0;
        text-decoration: underline;
    }
    .block-container {
        padding-top: 4rem !important;
    }
    div[data-testid="stTabs"] > div:first-child {
        position: sticky;
        top: 0;
        z-index: 999;
        background: white;
        padding-top: 4px;
        border-bottom: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── CSS TAMBAHAN: FONT COURIER & JARAK ANTAR SUB JUDUL ─────────
st.markdown("""
<style>
    html, body, [class*="css"], .stApp, .stMarkdown, p, span, div,
    .stDataFrame, .stMetric, .stButton button, .stTextInput input,
    .stSelectbox, .stNumberInput input, .stCaption, .stTabs {
        font-family: 'Courier New', Courier, monospace !important;
    }

    h1, h2, h3, h4 {
        font-family: 'Courier New', Courier, monospace !important;
        letter-spacing: 0.5px;
        font-weight: 700;
    }

    /* Ukuran font disesuaikan agar nyaman dibaca */
    h1 { font-size: 1.8rem !important; }
    h2 { font-size: 1.45rem !important; }
    h3 { font-size: 1.2rem !important; }

    p, span, label, div, .stMarkdown, .stCaption {
        font-size: 1.0rem !important;
        line-height: 1.6 !important;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        font-size: 0.88rem !important;
    }

    .stDataFrame, .stDataFrame * {
        font-size: 0.92rem !important;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
    }

    /* Jarak & garis pemisah tipis antar sub judul agar tidak rapat */
    div[data-testid="stVerticalBlock"] h3 {
        margin-top: 2rem;
        margin-bottom: 0.9rem;
        padding-bottom: 6px;
        border-bottom: 1px solid #eef2f7;
    }

    /* Sub judul pertama dalam sebuah tab tidak perlu jarak atas yang besar */
    div[data-testid="stTabContent"] > div > div[data-testid="stVerticalBlock"] > div:first-child h3 {
        margin-top: 0.4rem;
    }

    /* Tampilan tombol filter tahun (pills) */
    div[data-testid="stPills"] button {
        border-radius: 20px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ── CSS TAMBAHAN UNTUK HALAMAN LOGIN/LUPA PASSWORD/REGISTER ────
# Warna: biru gradasi (#0a3d62, #1a6fa8, #2980b9) dan putih saja
LOGIN_THEME_CSS = """
<style>
    /* Latar belakang halaman login tetap putih */
    .stApp {
        background: #ffffff;
    }
    /* Kotak form "Masuk" diberi gradasi biru */
    div[data-testid="stForm"] {
        background: linear-gradient(135deg, #0a3d62 0%, #1a6fa8 55%, #2980b9 100%);
        border-radius: 16px;
        padding: 28px 26px 20px 26px;
        box-shadow: 0 10px 32px rgba(10,61,98,0.25);
        border-top: 5px solid #aed6f1;
    }
    /* Teks label di dalam form (Email/Password) jadi putih agar kontras */
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] label,
    div[data-testid="stForm"] strong {
        color: #ffffff !important;
    }
    /* Kotak input tetap putih supaya teks isian mudah dibaca */
    div[data-testid="stForm"] input {
        background-color: #ffffff !important;
        color: #1a252f !important;
        border-radius: 8px !important;
        border: none !important;
    }
    /* Tombol submit utama (Masuk, Cari Password, Daftar): biru sedang */
    div[data-testid="stForm"] button[kind="primary"],
    .stButton button[kind="primary"] {
        background-color: #1a6fa8 !important;
        border-color: #1a6fa8 !important;
        color: #ffffff !important;
    }
    div[data-testid="stForm"] button[kind="primary"]:hover,
    .stButton button[kind="primary"]:hover {
        background-color: #0a3d62 !important;
        border-color: #0a3d62 !important;
    }
    /* Tombol sekunder (Lupa Password, Buat Akun Baru, Kembali): putih bersih */
    .stButton button:not([kind="primary"]) {
        border-color: #2980b9 !important;
        color: #0a3d62 !important;
        background-color: #ffffff !important;
    }
    .stButton button:not([kind="primary"]):hover {
        background-color: #eaf4fb !important;
        border-color: #0a3d62 !important;
    }
    .login-title, h1, h2, h3 {
        color: #0a3d62 !important;
    }
    /* Subtitle oranye diganti biru muda */
    div[style*="color:#f5821f"] {
        color: #aed6f1 !important;
    }
</style>
"""


# ══════════════════════════════════════════════════════════════
#  HALAMAN LOGIN
# ══════════════════════════════════════════════════════════════
def halaman_login():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; margin-bottom: 8px'>
            <span style='font-size:40px'>🚢</span>
        </div>
        <div style='text-align:center; font-size:20px; font-weight:700; color:#0a3d62; margin-bottom:4px'>
            PT Pelindo Marine Service
        </div>
        <div style='text-align:center; font-size:13px; color:#1a6fa8; font-weight:600; margin-bottom:28px'>
            Sistem Monitoring Arus Kas Operasional
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_login"):
            st.markdown("**Email**")
            email = st.text_input("Email", placeholder="Masukkan email", label_visibility="collapsed")
            st.markdown("**Password**")
            password = st.text_input("Password", type="password", placeholder="Masukkan password", label_visibility="collapsed")
            st.markdown("<div style='margin-top:4px'></div>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("Masuk", use_container_width=True, type="primary")

        if login_btn:
            accounts = st.session_state['accounts']
            if email == '':
                st.error("Email tidak boleh kosong.")
            elif email not in accounts:
                st.error("Email tidak terdaftar.")
            elif accounts[email] != password:
                st.error("Password salah.")
            else:
                st.session_state['logged_in']    = True
                st.session_state['current_user'] = email
                st.session_state['page']         = 'dashboard'
                st.rerun()

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Lupa Password?", use_container_width=True):
                st.session_state['page'] = 'forgot'
                st.rerun()
        with col_b:
            if st.button("Buat Akun Baru", use_container_width=True):
                st.session_state['page'] = 'register'
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  HALAMAN LUPA PASSWORD
# ══════════════════════════════════════════════════════════════
def halaman_lupa_password():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; font-size:36px; margin-bottom:8px'>🔑</div>
        <div style='text-align:center; font-size:20px; font-weight:700; color:#0a3d62; margin-bottom:4px'>
            Lupa Password
        </div>
        <div style='text-align:center; font-size:13px; color:#7f8c8d; margin-bottom:24px'>
            Masukkan email terdaftar untuk melihat password Anda
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_forgot"):
            st.markdown("**Email**")
            email_cek = st.text_input("Email", placeholder="Masukkan email terdaftar", label_visibility="collapsed")
            cek_btn   = st.form_submit_button("Cari Password", use_container_width=True, type="primary")

        if cek_btn:
            accounts = st.session_state['accounts']
            if email_cek == '':
                st.error("Email tidak boleh kosong.")
            elif email_cek not in accounts:
                st.error("Email tidak ditemukan. Pastikan email sudah terdaftar.")
            else:
                pw = accounts[email_cek]
                st.success(f"Password untuk **{email_cek}** adalah:")
                st.markdown(f"""
                <div style='background:#eaf4fb; border:1.5px solid #2980b9; border-radius:8px;
                            padding:14px 18px; font-size:20px; font-weight:700;
                            color:#0a3d62; text-align:center; letter-spacing:2px; margin-top:8px'>
                    {pw}
                </div>
                """, unsafe_allow_html=True)
                st.caption("Simpan password Anda di tempat yang aman.")

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("← Kembali ke Login", use_container_width=True):
            st.session_state['page'] = 'login'
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  HALAMAN DAFTAR AKUN BARU
# ══════════════════════════════════════════════════════════════
def halaman_register():
    col_l, col_m, col_r = st.columns([1, 1.4, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; font-size:36px; margin-bottom:8px'>📝</div>
        <div style='text-align:center; font-size:20px; font-weight:700; color:#0a3d62; margin-bottom:4px'>
            Buat Akun Baru
        </div>
        <div style='text-align:center; font-size:13px; color:#7f8c8d; margin-bottom:24px'>
            Daftarkan akun untuk mengakses dashboard
        </div>
        """, unsafe_allow_html=True)

        with st.form("form_register"):
            st.markdown("**Email**")
            email_baru = st.text_input("Email baru", placeholder="contoh@gmail.com", label_visibility="collapsed")
            st.markdown("**Password**")
            pass_baru  = st.text_input("Password baru", type="password",
                                        placeholder="Minimal 6 karakter", label_visibility="collapsed")
            st.markdown("**Konfirmasi Password**")
            pass_konfirm = st.text_input("Konfirmasi password", type="password",
                                          placeholder="Ulangi password", label_visibility="collapsed")
            daftar_btn = st.form_submit_button("Daftar", use_container_width=True, type="primary")

        if daftar_btn:
            accounts = st.session_state['accounts']
            if email_baru == '' or pass_baru == '':
                st.error("Email dan password tidak boleh kosong.")
            elif '@' not in email_baru or '.' not in email_baru:
                st.error("Format email tidak valid.")
            elif len(pass_baru) < 6:
                st.error("Password minimal 6 karakter.")
            elif pass_baru != pass_konfirm:
                st.error("Konfirmasi password tidak cocok.")
            elif email_baru in accounts:
                st.error("Email sudah terdaftar. Silakan login atau gunakan email lain.")
            else:
                st.session_state['accounts'][email_baru] = pass_baru
                st.success(f"Akun **{email_baru}** berhasil dibuat! Silakan login.")
                st.balloons()

        st.markdown("<div style='margin-top:16px'></div>", unsafe_allow_html=True)
        if st.button("← Kembali ke Login", use_container_width=True):
            st.session_state['page'] = 'login'
            st.rerun()


# ══════════════════════════════════════════════════════════════
#  FOOTER (ditambahkan untuk seluruh halaman dashboard)
# ══════════════════════════════════════════════════════════════
def tampilkan_footer():
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#7f8c8d; font-size:12px; padding:6px 0 18px 0'>"
        "Praktik Kerja Lapangan 2026 - Universitas Airlangga"
        "</div>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════
if not st.session_state['logged_in']:
    # Terapkan tema warna khusus (navy + putih) untuk halaman login/lupa password/register
    st.markdown(LOGIN_THEME_CSS, unsafe_allow_html=True)
    if st.session_state['page'] == 'forgot':
        halaman_lupa_password()
    elif st.session_state['page'] == 'register':
        halaman_register()
    else:
        halaman_login()
    tampilkan_footer()
    st.stop()

BULAN_ORDER = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
FEATURES    = ['arus_kas_operasi', 'pendapatan_operasi', 'beban_operasi']

def get_nama_klaster(k):
    if k == 2:
        return ['Performa Tinggi', 'Performa Rendah']
    elif k == 3:
        return ['Performa Tinggi', 'Performa Sedang', 'Performa Rendah']
    elif k == 4:
        return ['Performa Tinggi', 'Performa Sedang', 'Performa Cukup', 'Performa Rendah']
    elif k == 5:
        return ['Performa Tinggi', 'Performa Sedang Atas', 'Performa Sedang',
                'Performa Sedang Bawah', 'Performa Rendah']
    else:
        names = ['Performa Tinggi']
        for i in range(2, k):
            names.append(f'Performa Level {i}')
        names.append('Performa Rendah')
        return names

def get_colors(k):
    if k == 1:
        return {'Performa Tinggi': '#1a9e3f'}
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'klaster', ['#1a9e3f', '#f39c12', '#e74c3c'], N=k
    )
    nama = get_nama_klaster(k)
    return {nama[i]: to_hex(cmap(i / max(k-1, 1))) for i in range(k)}

@st.cache_data
def load_data(file, sheet):
    try:
        xl = pd.ExcelFile(file)
        if sheet not in xl.sheet_names:
            raise ValueError(
                f"Sheet **'{sheet}'** tidak ditemukan dalam file.\n\n"
                f"Sheet yang tersedia: **{', '.join(xl.sheet_names)}**"
            )

        df = pd.read_excel(file, sheet_name=sheet)

        required_cols = ['tahun', 'bulan', 'arus_kas_operasi', 'pendapatan_operasi', 'beban_operasi']
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Kolom berikut **tidak ditemukan** di sheet '{sheet}':\n"
                f"**{', '.join(missing_cols)}**\n\n"
                f"Kolom yang dibutuhkan: `{', '.join(required_cols)}`\n\n"
                f"Kolom yang ada di file: `{', '.join(df.columns.tolist())}`"
            )

        for col in ['arus_kas_operasi', 'pendapatan_operasi', 'beban_operasi']:
            df_test = df[df['tahun'] != 'Total'].copy()
            non_num = df_test[col].dropna()
            try:
                pd.to_numeric(non_num, errors='raise')
            except Exception:
                raise ValueError(
                    f"Kolom **'{col}'** mengandung nilai yang bukan angka. "
                    f"Pastikan kolom tersebut hanya berisi angka (juta Rp)."
                )

        df = df[df['tahun'] != 'Total'].copy()
        df = df[df['bulan'].notna()].copy()
        df['tahun']     = df['tahun'].ffill().astype(float).astype(int)
        df['bulan_num'] = df['bulan'].map({b: i+1 for i, b in enumerate(BULAN_ORDER)})

        invalid_bulan = df[df['bulan_num'].isna()]['bulan'].unique()
        if len(invalid_bulan) > 0:
            raise ValueError(
                f"Kolom **'bulan'** mengandung nilai tidak valid: **{', '.join(str(b) for b in invalid_bulan)}**\n\n"
                f"Nilai bulan yang valid: `{', '.join(BULAN_ORDER)}`"
            )

        if len(df) == 0:
            raise ValueError("File berhasil dibaca, tetapi **tidak ada baris data** yang valid setelah filter.")

        return df

    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Gagal membaca file Excel: **{str(e)}**\n\nPastikan file tidak rusak dan berformat `.xlsx`.")

def run_clustering(df, k):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)
    km       = KMeans(n_clusters=k, random_state=42, n_init=10)
    df       = df.copy()
    df['klaster_raw'] = km.fit_predict(X_scaled)
    means     = df.groupby('klaster_raw')['arus_kas_operasi'].mean()
    rank      = means.rank(ascending=False).astype(int)
    nama      = get_nama_klaster(k)
    label_map = {c: nama[r-1] for c, r in rank.items()}
    df['klaster'] = df['klaster_raw'].map(label_map)
    sil     = silhouette_score(X_scaled, km.labels_)
    inertia = km.inertia_
    return df, X_scaled, sil, inertia


# ══════════════════════════════════════════════════════════════
#  HELPER: NARASI ANALISIS HASIL CLUSTERING
# ══════════════════════════════════════════════════════════════
def generate_analysis_text(df_view, summary_view, sil, aktif_view):
    teks = []

    if sil >= 0.5:
        kualitas = "**baik** — klaster terbentuk dengan pemisahan yang cukup jelas antar kelompok"
    elif sil >= 0.25:
        kualitas = "**cukup baik**, meskipun terdapat sedikit tumpang tindih antar klaster"
    else:
        kualitas = "**kurang optimal** — klaster cenderung tumpang tindih sehingga interpretasi perlu dilakukan dengan kehati-hatian"

    teks.append(f"Nilai **Silhouette Score = {sil:.5f}** menunjukkan kualitas pengelompokan tergolong {kualitas}.")
    teks.append("")

    teks.append("**Karakteristik tiap klaster (berdasarkan tahun yang dipilih):**")
    for kl in aktif_view:
        if kl not in summary_view.index:
            continue
        row   = summary_view.loc[kl]
        sub   = df_view[df_view['klaster'] == kl]
        jumlah = len(sub)
        persen = jumlah / len(df_view) * 100 if len(df_view) > 0 else 0
        tahun_count = sub['tahun'].value_counts()
        tahun_top   = ", ".join(str(t) for t in tahun_count.index[:3])
        teks.append(
            f"- **{kl}**: {jumlah} bulan ({persen:.1f}% dari data terpilih). "
            f"Rata-rata arus kas operasi **Rp {row['Rata-rata Arus Kas']:,.2f} juta**, "
            f"pendapatan operasi **Rp {row['Rata-rata Pendapatan']:,.2f} juta**, "
            f"beban operasi **Rp {row['Rata-rata Beban']:,.2f} juta**. "
            f"Paling banyak terjadi pada tahun {tahun_top}."
        )

    teks.append("")
    teks.append("**Klaster dominan per tahun (pada rentang tahun yang dipilih):**")
    for yr in sorted(df_view['tahun'].unique()):
        dom = df_view[df_view['tahun'] == yr]['klaster'].value_counts().idxmax()
        teks.append(f"- Tahun {yr}: didominasi oleh klaster **{dom}**")

    if 'Performa Rendah' in summary_view.index and 'Performa Tinggi' in summary_view.index:
        tinggi = summary_view.loc['Performa Tinggi']
        rendah = summary_view.loc['Performa Rendah']
        teks.append("")
        teks.append(
            f"**Catatan manajerial:** Selisih rata-rata arus kas operasi antara klaster *Performa Tinggi* "
            f"(Rp {tinggi['Rata-rata Arus Kas']:,.2f} juta) dan *Performa Rendah* "
            f"(Rp {rendah['Rata-rata Arus Kas']:,.2f} juta) cukup signifikan. "
            f"Bulan-bulan yang masuk klaster *Performa Rendah* sebaiknya dievaluasi lebih lanjut, "
            f"khususnya proporsi beban operasi terhadap pendapatan operasi pada periode tersebut, "
            f"agar dapat diidentifikasi penyebab rendahnya arus kas dan langkah perbaikan yang diperlukan."
        )

    return "\n".join(teks)


# ══════════════════════════════════════════════════════════════
#  HELPER: FILTER TAHUN BERGAYA TOMBOL (PILLS) DENGAN FALLBACK
# ══════════════════════════════════════════════════════════════
def filter_tahun_pills(label, options, default, key):
    """Filter tahun ala tombol klik (st.pills). Jika versi Streamlit
    belum mendukung st.pills, otomatis jatuh ke st.multiselect."""
    if hasattr(st, "pills"):
        return st.pills(
            label,
            options=options,
            default=default,
            selection_mode="multi",
            key=key
        )
    else:
        return st.multiselect(label, options=options, default=default, key=key)


# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='background:linear-gradient(160deg,#0a3d62,#1a6fa8);
                border-radius:10px; padding:20px 16px; margin-bottom:4px; text-align:center'>
        <div style='font-size:36px; margin-bottom:8px'>🚢</div>
        <div style='font-size:11px; color:#aed6f1; letter-spacing:2px; font-weight:700; margin-bottom:4px'>
            PT PELINDO MARINE SERVICE
        </div>
        <div style='width:40px; height:2px; background:#aed6f1; margin:8px auto'></div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.header("⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    sheet_name    = st.text_input("Nama sheet", value="bulanan")
    K             = st.number_input("Jumlah Klaster (K)", min_value=2, max_value=8, value=4)

    st.divider()
    st.markdown("**Preview Klaster**")
    COLORS = get_colors(K)
    NAMA_K = get_nama_klaster(K)
    for nm in NAMA_K:
        warna = COLORS[nm]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'<div style="width:14px;height:14px;border-radius:3px;background:{warna}"></div>'
            f'<span style="font-size:13px">{nm}</span></div>',
            unsafe_allow_html=True
        )
    st.divider()

# ── BANNER ───────────────────────────────────────────────────
BANNER_HTML = """
<div style="background:linear-gradient(135deg,#0a3d62 0%,#1a6fa8 60%,#2980b9 100%);
            padding:28px 36px; border-radius:12px; margin-bottom:24px;
            display:flex; align-items:center; gap:24px">
    <div>
        <div style="font-size:12px; color:#aed6f1; font-weight:700; letter-spacing:2px; margin-bottom:4px">
            PT PELINDO MARINE SERVICE
        </div>
        <div style="font-size:22px; font-weight:800; color:white; line-height:1.3">
            Sistem Monitoring Arus Kas Operasional
        </div>
        <div style="font-size:13px; color:#aed6f1; margin-top:6px">
            Berbasis K-Means Clustering &amp; Dashboard Interaktif
        </div>
    </div>
</div>
"""

st.markdown(BANNER_HTML, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab_manual, tab_logout = st.tabs([
    "Ringkasan Data",
    "Penentuan K Optimal",
    "Hasil Clustering",
    "Visualisasi",
    "Export",
    "Input Manual",
    "Logout"
])

# ═══════════════════════════════════════════════════════════════
# TAB INPUT MANUAL
# ═══════════════════════════════════════════════════════════════
with tab_manual:
    st.subheader("📥 Input Data Manual")
    st.markdown("Isi tabel di bawah sesuai data keuangan. Data ini bisa langsung digunakan untuk analisis clustering.")

    col_r, col_k = st.columns([1, 1])
    with col_r:
        n_baris = st.number_input("Jumlah baris (bulan)", min_value=1, max_value=200, value=12, step=1, key="n_baris")
    with col_k:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.caption(f"Tabel akan memiliki **{n_baris} baris** × **5 kolom** (Tahun, Bulan, Arus Kas, Pendapatan, Beban)")

    if st.session_state['manual_df'] is not None and len(st.session_state['manual_df']) == n_baris:
        default_data = st.session_state['manual_df']
    else:
        bulan_cycle = BULAN_ORDER * (n_baris // 12 + 1)
        tahun_list  = []
        bulan_list  = []
        tahun_awal  = 2018
        for i in range(n_baris):
            tahun_list.append(tahun_awal + i // 12)
            bulan_list.append(bulan_cycle[i % 12])
        default_data = pd.DataFrame({
            'tahun':              tahun_list,
            'bulan':              bulan_list,
            'arus_kas_operasi':   [0.0] * n_baris,
            'pendapatan_operasi': [0.0] * n_baris,
            'beban_operasi':      [0.0] * n_baris,
        })

    st.markdown("**Isi tabel berikut (satuan: juta Rp):**")
    edited_df = st.data_editor(
        default_data,
        use_container_width=True,
        num_rows="fixed",
        column_config={
            "tahun": st.column_config.NumberColumn(
                "Tahun", min_value=2000, max_value=2100, step=1, format="%d"
            ),
            "bulan": st.column_config.SelectboxColumn(
                "Bulan", options=BULAN_ORDER
            ),
            "arus_kas_operasi": st.column_config.NumberColumn(
                "Arus Kas Operasi (juta Rp)", format="%.2f"
            ),
            "pendapatan_operasi": st.column_config.NumberColumn(
                "Pendapatan Operasi (juta Rp)", format="%.2f"
            ),
            "beban_operasi": st.column_config.NumberColumn(
                "Beban Operasi (juta Rp)", format="%.2f"
            ),
        },
        key="editor_manual"
    )

    col_s, col_g, col_r2 = st.columns([1, 1, 1])
    with col_s:
        if st.button("💾 Simpan & Gunakan Data Ini", use_container_width=True, type="primary"):
            if edited_df[FEATURES].isnull().any().any():
                st.error("Ada nilai kosong. Pastikan semua sel terisi.")
            elif (edited_df[FEATURES] == 0).all().all():
                st.warning("Semua nilai masih 0. Silakan isi data terlebih dahulu.")
            else:
                df_manual = edited_df.copy()
                df_manual['tahun']     = df_manual['tahun'].astype(int)
                df_manual['bulan_num'] = df_manual['bulan'].map({b: i+1 for i, b in enumerate(BULAN_ORDER)})
                st.session_state['manual_df']  = df_manual
                st.session_state['use_manual'] = True
                st.success("✅ Data manual berhasil disimpan dan akan digunakan untuk analisis!")
                st.rerun()
    with col_g:
        if st.button("🗑️ Reset Data Manual", use_container_width=True):
            st.session_state['manual_df']  = None
            st.session_state['use_manual'] = False
            st.rerun()
    with col_r2:
        buf_tmpl = io.BytesIO()
        template = pd.DataFrame({
            'tahun': [''] * 12,
            'bulan': BULAN_ORDER,
            'arus_kas_operasi': [''] * 12,
            'pendapatan_operasi': [''] * 12,
            'beban_operasi': [''] * 12,
        })
        with pd.ExcelWriter(buf_tmpl, engine='openpyxl') as w:
            template.to_excel(w, index=False, sheet_name='bulanan')
        buf_tmpl.seek(0)
        st.download_button(
            "📄 Download Template Excel",
            data=buf_tmpl,
            file_name="template_input.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

    if st.session_state['use_manual'] and st.session_state['manual_df'] is not None:
        st.success("✅ Data manual **aktif** — tab Ringkasan, Clustering, dan Visualisasi menggunakan data ini.")
        st.dataframe(st.session_state['manual_df'], use_container_width=True, height=250)
    elif uploaded_file is not None:
        st.info("ℹ️ Saat ini menggunakan data dari file Excel yang di-upload.")

# ── Tentukan sumber data ──────────────────────────────────────
use_manual_data = st.session_state['use_manual'] and st.session_state['manual_df'] is not None
data_loaded     = uploaded_file is not None or use_manual_data

if not data_loaded:
    for t in [tab1, tab2, tab3, tab4, tab5]:
        with t:
            st.info("Upload file Excel di sidebar **atau** isi data manual di tab **Input Manual** untuk memulai analisis.")
    with tab_logout:
        col_l, col_m, col_r = st.columns([1, 1.2, 1])
        with col_m:
            st.markdown("<div style='text-align:center;font-size:48px;margin:24px 0'>🚪</div>", unsafe_allow_html=True)
            st.markdown("<div style='text-align:center;font-size:18px;font-weight:600;color:#2c3e50;margin-bottom:8px'>Yakin ingin keluar?</div>", unsafe_allow_html=True)
            if st.button("Ya, Logout", use_container_width=True, type="primary"):
                st.session_state['logged_in']    = False
                st.session_state['current_user'] = ""
                st.session_state['page']         = "login"
                st.rerun()
    tampilkan_footer()
    st.stop()

# ── Load data ─────────────────────────────────────────────────
if use_manual_data:
    df_raw = st.session_state['manual_df'].copy()
else:
    try:
        df_raw = load_data(uploaded_file, sheet_name)
    except ValueError as e:
        st.error(f"⚠️ **Format file tidak sesuai**\n\n{str(e)}")
        st.markdown("""
        **Panduan format file yang benar:**
        - Format file: `.xlsx`
        - Nama sheet sesuai yang diisi di sidebar (default: `bulanan`)
        - Kolom wajib: `tahun`, `bulan`, `arus_kas_operasi`, `pendapatan_operasi`, `beban_operasi`
        - Kolom `bulan` harus berisi: `Jan`, `Feb`, `Mar`, `Apr`, `Mei`, `Jun`, `Jul`, `Agu`, `Sep`, `Okt`, `Nov`, `Des`
        - Kolom angka harus berisi nilai numerik (juta Rp)
        
        💡 Download template di tab **Input Manual → Download Template Excel** sebagai referensi format.
        """)
        tampilkan_footer()
        st.stop()

df, X_scaled, sil, inertia_terpilih = run_clustering(df_raw, K)
COLORS = get_colors(K)
NAMA_K = get_nama_klaster(K)
aktif  = [n for n in NAMA_K if n in df['klaster'].values]


# ═══════════════════════════════════════════════════════════════
# TAB 1 — RINGKASAN DATA
# ═══════════════════════════════════════════════════════════════
with tab1:
    if use_manual_data:
        st.info("📋 Menggunakan **data manual** yang diinput di tab Input Manual.")
    st.subheader("Ringkasan Data")

    # ── FILTER TAHUN (klik/pencet) ───────────────────────────────
    st.markdown("**Filter Tahun**")
    tahun_options_t1 = sorted(df['tahun'].unique().tolist())
    tahun_pilihan_t1 = filter_tahun_pills(
        "Pilih tahun yang ingin ditampilkan",
        options=tahun_options_t1,
        default=tahun_options_t1,
        key="filter_tahun_ringkasan"
    )

    if not tahun_pilihan_t1:
        st.warning("⚠️ Pilih minimal satu tahun untuk menampilkan ringkasan data.")
        df_t1 = df.iloc[0:0].copy()
    else:
        df_t1 = df[df['tahun'].isin(tahun_pilihan_t1)].copy()

    if len(df_t1) == 0:
        st.info("Tidak ada data untuk tahun yang dipilih.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Observasi",  f"{len(df_t1)} bulan")
        c2.metric("Periode",          f"{df_t1['tahun'].min()} – {df_t1['tahun'].max()}")
        c3.metric("Missing Values",   int(df_t1[FEATURES].isnull().sum().sum()))
        c4.metric("Jumlah Tahun",     df_t1['tahun'].nunique())

        st.subheader("Statistik Deskriptif (juta Rp)")
        desc = df_t1[FEATURES].describe().round(5)
        desc.index = ['Jumlah','Rata-rata','Std Dev','Min','Q1','Median','Q3','Maks']
        st.dataframe(desc, use_container_width=True)

        st.subheader("Koefisien Variasi (%)")
        cv = {f: round((df_t1[f].std()/df_t1[f].mean())*100, 5) for f in FEATURES}
        st.dataframe(pd.DataFrame(cv, index=['CV (%)']).T.rename_axis("Variabel"),
                     use_container_width=True)

        st.subheader("Uji Normalitas Shapiro-Wilk (α=0.05)")
        if len(df_t1) >= 3:
            rows = []
            for f in FEATURES:
                stat, p = stats.shapiro(df_t1[f])
                rows.append({'Variabel': f, 'Statistik': round(stat, 5),
                             'p-value': round(p, 5),
                             'Distribusi': 'Normal' if p > 0.05 else 'Tidak Berdistribusi Normal'})
            df_norm = pd.DataFrame(rows).set_index('Variabel')
            st.dataframe(df_norm, use_container_width=True)
            # Cek apakah semua distribusi Normal
            all_normal = all(r['Distribusi'] == 'Normal' for r in rows)
            if all_normal:
                st.caption("→ Semua variabel berdistribusi normal → normalisasi Z-Score tetap digunakan sebelum clustering untuk menyamakan skala.")
            else:
                st.caption("→ Data tidak berdistribusi normal → normalisasi Z-Score tepat digunakan sebelum clustering.")
        else:
            st.info("Data terlalu sedikit (minimal 3 bulan) untuk melakukan uji Shapiro-Wilk.")

        # ── TAMBAHAN 1: HASIL NORMALISASI Z-SCORE ────────────────
        st.subheader("Hasil Normalisasi Z-Score")
        st.caption(
            "Z-Score = (X − rata-rata) / standar deviasi. "
            "Nilai 0 berarti sama dengan rata-rata, nilai positif berarti di atas rata-rata, "
            "dan nilai negatif berarti di bawah rata-rata. Data inilah yang digunakan sebagai "
            "input proses K-Means Clustering."
        )
        scaler_t1   = StandardScaler()
        X_scaled_t1 = scaler_t1.fit_transform(df_t1[FEATURES].values)
        df_zscore = pd.DataFrame(
            X_scaled_t1,
            columns=[f"{f}_zscore" for f in FEATURES],
            index=df_t1.index
        ).round(5)
        df_zscore.insert(0, 'bulan', df_t1['bulan'].values)
        df_zscore.insert(0, 'tahun', df_t1['tahun'].values)
        st.dataframe(df_zscore, use_container_width=True, height=300)

        st.subheader("Rasio Arus Kas / Pendapatan per Tahun (%)")
        rasio = df_t1.groupby('tahun').apply(
            lambda g: round(g['arus_kas_operasi'].sum() / g['pendapatan_operasi'].sum() * 100, 5)
        ).reset_index(name='Rasio (%)')
        rasio['Keterangan'] = rasio['Rasio (%)'].apply(lambda r: '⚠️ ANOMALI' if r < 2 else '✅ Normal')
        st.dataframe(rasio.set_index('tahun'), use_container_width=True)

        st.subheader("Preview Data")
        st.dataframe(df_raw[df_raw['tahun'].isin(tahun_pilihan_t1)], use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PENENTUAN K OPTIMAL
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Penentuan K Optimal")

    n_samples = len(df_raw)

    # Validasi: data minimal harus ada cukup baris untuk K
    if n_samples < 4:
        st.warning(f"⚠️ Data terlalu sedikit ({n_samples} baris) untuk analisis K optimal. Minimal diperlukan 4 baris data.")
        st.stop()

    scaler_opt = StandardScaler()
    Xs_opt     = scaler_opt.fit_transform(df_raw[FEATURES].values)

    # Batasi K_range agar tidak melebihi jumlah sampel
    max_k      = min(8, n_samples - 1)
    K_range    = range(2, max_k + 1)

    if len(list(K_range)) < 1:
        st.warning("⚠️ Tidak cukup variasi K untuk dievaluasi. Tambahkan lebih banyak data.")
        st.stop()

    inertia_list = []
    sil_scores   = []

    with st.spinner("Menghitung metrik untuk tiap K..."):
        for k in K_range:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(Xs_opt)
            inertia_list.append(km.inertia_)
            # FIX: tangkap error silhouette_score jika klaster tidak valid
            try:
                sil_val = silhouette_score(Xs_opt, labels)
            except ValueError:
                sil_val = 0.0
            sil_scores.append(sil_val)

    # Deteksi elbow otomatis
    ia   = np.array(inertia_list)
    kv   = np.array(list(K_range))
    p1, p2 = np.array([kv[0], ia[0]]), np.array([kv[-1], ia[-1]])
    dists  = [np.abs(np.cross(p2-p1, p1-np.array([kv[i], ia[i]]))) / np.linalg.norm(p2-p1)
              for i in range(len(kv))]
    elbow_k    = int(kv[np.argmax(dists)])
    sil_best_k = list(K_range)[sil_scores.index(max(sil_scores))]

    # Tabel metrik dengan 5 desimal
    metrics_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia (SSE)': [round(v, 5) for v in inertia_list],
        'Silhouette Score': [round(v, 5) for v in sil_scores],
    }).set_index('K')
    st.dataframe(metrics_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Elbow terdeteksi",     f"K = {elbow_k}")
    c2.metric("Silhouette tertinggi", f"K = {sil_best_k} ({max(sil_scores):.5f})")
    c3.metric("K yang digunakan",     f"K = {K}")

    # Grafik: Elbow + Silhouette (2 panel)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#FAFAFA')

    axes[0].plot(list(K_range), inertia_list, 'o-', color='#2980b9', linewidth=2.5, markersize=8)
    axes[0].axvline(x=elbow_k, color='#27ae60', linestyle='--', linewidth=1.8,
                    label=f'Elbow = {elbow_k}')
    axes[0].axvline(x=K, color='#e74c3c', linestyle='--', linewidth=2,
                    label=f'K terpilih = {K}')
    axes[0].set_title('Elbow Method (Inertia / SSE)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Jumlah Klaster (K)')
    axes[0].set_ylabel('Inertia')
    axes[0].legend(fontsize=10)
    axes[0].grid(alpha=0.3)
    axes[0].set_facecolor('#FAFAFA')

    axes[1].plot(list(K_range), sil_scores, 's-', color='#8e44ad', linewidth=2.5, markersize=8)
    axes[1].axvline(x=sil_best_k, color='#27ae60', linestyle='--', linewidth=1.8,
                    label=f'Terbaik = {sil_best_k}')
    axes[1].axvline(x=K, color='#e74c3c', linestyle='--', linewidth=2,
                    label=f'K terpilih = {K}')
    axes[1].set_title('Silhouette Score', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Jumlah Klaster (K)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].legend(fontsize=10)
    axes[1].grid(alpha=0.3)
    axes[1].set_facecolor('#FAFAFA')

    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("📝 Interpretasi pemilihan K"):
        idx_K   = list(K_range).index(K) if K in K_range else 0
        selisih = round(max(sil_scores) - sil_scores[idx_K], 5)
        st.markdown(f"""
- **Elbow Method** menunjukkan siku paling jelas pada **K = {elbow_k}** (inertia mulai landai di titik ini).
- **Silhouette Score** tertinggi ada di **K = {sil_best_k}** ({max(sil_scores):.5f}), namun K = {K} ({sil_scores[idx_K]:.5f}) tidak berbeda jauh (selisih {selisih:.5f}).
- **K = {K}** dipilih karena menghasilkan {K} klaster yang dapat diinterpretasikan secara manajerial: {", ".join(NAMA_K)}.
        """)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — HASIL CLUSTERING
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Hasil K-Means Clustering (K={K})")

    c1, c2 = st.columns(2)
    c1.metric("Silhouette Score", f"{sil:.5f}", help="Mendekati 1 = sangat baik")
    c2.metric("Elbow — Inertia (SSE)", f"{inertia_terpilih:,.5f}",
              help="Jumlah kuadrat jarak tiap titik ke pusat klasternya; makin kecil makin kompak")

    # ── TAMBAHAN 2: FILTER TAHUN UNTUK MELIHAT HASIL CLUSTERING ──
    st.subheader("🗂️ Filter Tahun")
    tahun_options = sorted(df['tahun'].unique().tolist())
    tahun_pilihan = filter_tahun_pills(
        "Pilih tahun yang ingin ditampilkan pada hasil clustering di bawah ini",
        options=tahun_options,
        default=tahun_options,
        key="filter_tahun_clustering"
    )

    if not tahun_pilihan:
        st.warning("⚠️ Pilih minimal satu tahun untuk menampilkan hasil clustering.")
        df_view = df.iloc[0:0].copy()
    else:
        df_view = df[df['tahun'].isin(tahun_pilihan)].copy()

    aktif_view = [n for n in NAMA_K if n in df_view['klaster'].values] if len(df_view) > 0 else []

    st.subheader("Statistik Per Klaster (rata-rata, juta Rp)")
    if len(df_view) > 0:
        summary_view = df_view.groupby('klaster')[FEATURES].mean().round(5)
        summary_view.columns = ['Rata-rata Arus Kas', 'Rata-rata Pendapatan', 'Rata-rata Beban']
        summary_view = summary_view.reindex([n for n in NAMA_K if n in summary_view.index])
        st.dataframe(summary_view, use_container_width=True)
    else:
        summary_view = pd.DataFrame(columns=['Rata-rata Arus Kas', 'Rata-rata Pendapatan', 'Rata-rata Beban'])
        st.info("Tidak ada data untuk tahun yang dipilih.")

    st.subheader("Jumlah Bulan Per Klaster")
    if len(df_view) > 0:
        cnt = df_view['klaster'].value_counts().reindex(aktif_view).rename('Jumlah Bulan').to_frame()
        st.dataframe(cnt, use_container_width=True)
    else:
        st.info("Tidak ada data.")

    st.subheader("Distribusi Klaster per Tahun")
    if len(df_view) > 0:
        cross_view = df_view.groupby(['tahun','klaster']).size().unstack(fill_value=0)
        cross_view = cross_view.reindex(columns=[n for n in NAMA_K if n in cross_view.columns])
        st.dataframe(cross_view, use_container_width=True)
    else:
        st.info("Tidak ada data.")


    if 2023 in df['tahun'].values:
        st.subheader("Analisis Anomali 2023")
        df23   = df[df['tahun']==2023]
        r23    = df23['arus_kas_operasi'].mean() / df23['pendapatan_operasi'].mean() * 100
        r_else = df[df['tahun']!=2023]['arus_kas_operasi'].mean() / \
                 df[df['tahun']!=2023]['pendapatan_operasi'].mean() * 100
        c1, c2 = st.columns(2)
        c1.metric("Rasio Arus Kas/Pendapatan 2023",    f"{r23:.5f}%")
        c2.metric("Rasio Arus Kas/Pendapatan Non-2023", f"{r_else:.5f}%")
        if r23 < 2:
            st.warning("⚠️ Tahun 2023 menunjukkan anomali: pendapatan tertinggi namun arus kas operasi sangat rendah.")

    st.subheader("Data Hasil Clustering")
    st.dataframe(df_view[['tahun','bulan','arus_kas_operasi','pendapatan_operasi',
                      'beban_operasi','klaster']], use_container_width=True, height=350)

    # ── TAMBAHAN 3: ANALISIS HASIL CLUSTERING ────────────────────
    st.subheader("📊 Analisis Hasil Clustering")
    if len(df_view) > 0:
        st.markdown(generate_analysis_text(df_view, summary_view, sil, aktif_view))
    else:
        st.info("Tidak ada data untuk dianalisis pada tahun yang dipilih.")


# ═══════════════════════════════════════════════════════════════
# TAB 4 — VISUALISASI
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Visualisasi Hasil Clustering")

    df_s = df.sort_values(['tahun','bulan_num']).reset_index(drop=True)

    st.markdown("#### Tren Arus Kas Operasi Bulanan")
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    fig1.patch.set_facecolor('#FAFAFA')
    for i in range(len(df_s)-1):
        c = COLORS.get(df_s.iloc[i]['klaster'], '#999')
        ax1.plot([i, i+1],
                 [df_s.iloc[i]['arus_kas_operasi'], df_s.iloc[i+1]['arus_kas_operasi']],
                 color=c, linewidth=1.5, alpha=0.5, zorder=1)
    for kl in aktif:
        grp = df_s[df_s['klaster']==kl]
        ax1.scatter(grp.index, grp['arus_kas_operasi'],
                    c=COLORS[kl], label=kl, s=60, zorder=3,
                    edgecolors='white', linewidth=0.8)
    for yr in df_s['tahun'].unique():
        xi = df_s[df_s['tahun']==yr].index[0]
        ax1.axvline(x=xi, color='#aaa', linewidth=0.7, linestyle=':', alpha=0.6)
        ax1.text(xi+0.3, df_s['arus_kas_operasi'].max()*0.97, str(yr), fontsize=8, color='#555')
    ax1.set_ylabel('Arus Kas (juta Rp)', fontsize=9)
    ax1.set_xticks([])
    ax1.legend(fontsize=8, loc='upper left', ncol=min(K,3))
    ax1.grid(axis='y', alpha=0.25)
    ax1.set_facecolor('white')
    plt.tight_layout()
    st.pyplot(fig1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Pendapatan vs Arus Kas")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        fig2.patch.set_facecolor('#FAFAFA')
        for kl in aktif:
            g = df[df['klaster']==kl]
            ax2.scatter(g['pendapatan_operasi']/1000, g['arus_kas_operasi']/1000,
                        c=COLORS[kl], label=kl, alpha=0.85, s=60,
                        edgecolors='white', linewidth=0.6)
        ax2.set_xlabel('Pendapatan (miliar Rp)', fontsize=9)
        ax2.set_ylabel('Arus Kas (miliar Rp)', fontsize=9)
        ax2.legend(fontsize=7)
        ax2.grid(alpha=0.25)
        ax2.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig2)

    with c2:
        st.markdown("#### Distribusi Klaster per Tahun")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        fig3.patch.set_facecolor('#FAFAFA')
        cross_t = df.groupby(['tahun','klaster']).size().unstack(fill_value=0)
        cols_ok = [n for n in NAMA_K if n in cross_t.columns]
        cross_t[cols_ok].plot(kind='bar', ax=ax3,
                              color=[COLORS[c] for c in cols_ok],
                              edgecolor='white', width=0.7)
        ax3.set_xlabel('Tahun', fontsize=9)
        ax3.set_ylabel('Jumlah Bulan', fontsize=9)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=7)
        ax3.grid(axis='y', alpha=0.25)
        ax3.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig3)

    st.markdown("#### Heatmap Klaster Per Bulan Per Tahun")
    klaster_num_map   = {kl: i for i, kl in enumerate(aktif)}
    df['klaster_num'] = df['klaster'].map(klaster_num_map)
    pivot = df.pivot_table(index='tahun', columns='bulan', values='klaster_num', aggfunc='first')
    pivot = pivot[[b for b in BULAN_ORDER if b in pivot.columns]]

    cmap  = ListedColormap([COLORS[kl] for kl in aktif])
    fig4, ax4 = plt.subplots(figsize=(13, max(3, len(pivot)*0.6+1)))
    fig4.patch.set_facecolor('#FAFAFA')
    ax4.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-0.5, vmax=len(aktif)-0.5)
    ax4.set_xticks(range(len(pivot.columns)))
    ax4.set_xticklabels(pivot.columns.tolist(), fontsize=10)
    ax4.set_yticks(range(len(pivot.index)))
    ax4.set_yticklabels(pivot.index.tolist(), fontsize=10)
    patches = [mpatches.Patch(color=COLORS[kl], label=kl) for kl in aktif]
    ax4.legend(handles=patches, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=9)
    inisial_map = {kl: ''.join(w[0] for w in kl.split()) for kl in aktif}
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                label = inisial_map.get(aktif[int(val)], '?')
                ax4.text(j, i, label, ha='center', va='center',
                         fontsize=9, fontweight='bold', color='white')
    plt.tight_layout()
    st.pyplot(fig4)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — EXPORT
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Export Hasil")

    # ── FILTER TAHUN UNTUK EXPORT ────────────────────────────────
    st.markdown("**Filter Tahun**")
    tahun_options_export = sorted(df['tahun'].unique().tolist())
    tahun_pilihan_export = filter_tahun_pills(
        "Pilih tahun yang ingin di-export",
        options=tahun_options_export,
        default=tahun_options_export,
        key="filter_tahun_export"
    )

    out_cols = ['tahun','bulan','arus_kas_operasi','pendapatan_operasi','beban_operasi','klaster']

    if not tahun_pilihan_export:
        st.warning("⚠️ Pilih minimal satu tahun untuk melakukan export.")
    else:
        out_df = df[df['tahun'].isin(tahun_pilihan_export)][out_cols].copy()

        st.caption(f"Menampilkan **{len(out_df)} baris** dari tahun yang dipilih: {', '.join(str(t) for t in sorted(tahun_pilihan_export))}")
        st.dataframe(out_df, use_container_width=True)

        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            out_df.to_excel(writer, index=False, sheet_name='Hasil Clustering')
        buf.seek(0)

        st.download_button(
            label="⬇️ Download hasil_clustering.xlsx",
            data=buf,
            file_name="hasil_clustering.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


# ═══════════════════════════════════════════════════════════════
# TAB LOGOUT
# ═══════════════════════════════════════════════════════════════
with tab_logout:
    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    col_l, col_m, col_r = st.columns([1, 1.2, 1])
    with col_m:
        st.markdown("""
        <div style='text-align:center; font-size:48px; margin-bottom:12px'>🚪</div>
        <div style='text-align:center; font-size:18px; font-weight:600; color:#2c3e50; margin-bottom:8px'>
            Yakin ingin keluar?
        </div>
        <div style='text-align:center; font-size:13px; color:#7f8c8d; margin-bottom:24px'>
            Sesi Anda akan diakhiri dan Anda akan kembali ke halaman login.
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚪 Ya, Logout", key="logout_main", use_container_width=True, type="primary"):
            st.session_state['logged_in']    = False
            st.session_state['current_user'] = ''
            st.session_state['page']         = 'login'
            st.rerun()

# ── TAMBAHAN 4: FOOTER DASHBOARD ──────────────────────────────
tampilkan_footer()
