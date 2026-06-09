import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_hex
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
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

BULAN_ORDER = ['Jan','Feb','Mar','Apr','Mei','Jun','Jul','Agu','Sep','Okt','Nov','Des']
FEATURES    = ['arus_kas_operasi', 'pendapatan_operasi', 'beban_operasi']

# ── FUNGSI: NAMA KLASTER DINAMIS ─────────────────────────────
def get_nama_klaster(k):
    """Hasilkan nama klaster dari tertinggi ke terendah sesuai K."""
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
        # K >= 6: Tinggi, Level 2, Level 3, ..., Rendah
        names = ['Performa Tinggi']
        for i in range(2, k):
            names.append(f'Performa Level {i}')
        names.append('Performa Rendah')
        return names

# ── FUNGSI: WARNA GRADASI DINAMIS ────────────────────────────
def get_colors(k):
    """Gradasi hijau → kuning → merah sesuai jumlah klaster."""
    if k == 1:
        return {'Performa Tinggi': '#1a9e3f'}
    # Buat gradasi dari hijau (#1a9e3f) → kuning (#f39c12) → merah (#e74c3c)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'klaster', ['#1a9e3f', '#f39c12', '#e74c3c'], N=k
    )
    nama = get_nama_klaster(k)
    return {nama[i]: to_hex(cmap(i / max(k-1, 1))) for i in range(k)}

# ── LOAD DATA ─────────────────────────────────────────────────
@st.cache_data
def load_data(file, sheet):
    df = pd.read_excel(file, sheet_name=sheet)
    df = df[df['tahun'] != 'Total'].copy()
    df = df[df['bulan'].notna()].copy()
    df['tahun']    = df['tahun'].ffill().astype(float).astype(int)
    df['bulan_num'] = df['bulan'].map({b: i+1 for i, b in enumerate(BULAN_ORDER)})
    return df

# ── FUNGSI: JALANKAN CLUSTERING ───────────────────────────────
def run_clustering(df, k):
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].values)
    km       = KMeans(n_clusters=k, random_state=42, n_init=10)
    df       = df.copy()
    df['klaster_raw'] = km.fit_predict(X_scaled)

    # Urutkan klaster berdasarkan rata-rata arus kas (tertinggi = klaster 0)
    means    = df.groupby('klaster_raw')['arus_kas_operasi'].mean()
    rank     = means.rank(ascending=False).astype(int)
    nama     = get_nama_klaster(k)
    label_map = {c: nama[r-1] for c, r in rank.items()}
    df['klaster'] = df['klaster_raw'].map(label_map)

    sil = silhouette_score(X_scaled, km.labels_)
    db  = davies_bouldin_score(X_scaled, km.labels_)
    return df, X_scaled, sil, db

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    sheet_name    = st.text_input("Nama sheet", value="bulanan")
    K             = st.slider("Jumlah Klaster (K)", min_value=2, max_value=8, value=3)

    # Preview nama & warna klaster
    st.divider()
    st.markdown("**Preview Klaster**")
    COLORS     = get_colors(K)
    NAMA_K     = get_nama_klaster(K)
    for nm in NAMA_K:
        warna = COLORS[nm]
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px">'
            f'<div style="width:14px;height:14px;border-radius:3px;background:{warna}"></div>'
            f'<span style="font-size:13px">{nm}</span></div>',
            unsafe_allow_html=True
        )
    st.divider()
    st.caption("PKL — Analisis Kinerja Keuangan")

# ── HEADER ───────────────────────────────────────────────────
st.title("🚢 Analisis K-Means Clustering")
st.markdown("**PT Pelindo Marine Service 2018–2024** — Kinerja Keuangan Operasional")
st.divider()

if uploaded_file is None:
    st.info("👈 Upload file Excel di sidebar untuk memulai analisis.")
    st.stop()

df_raw = load_data(uploaded_file, sheet_name)
df, X_scaled, sil, db = run_clustering(df_raw, K)
COLORS  = get_colors(K)
NAMA_K  = get_nama_klaster(K)
aktif   = [n for n in NAMA_K if n in df['klaster'].values]

# ── TABS ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Ringkasan Data",
    "🔍 Penentuan K Optimal",
    "🎯 Hasil Clustering",
    "📈 Visualisasi",
    "🔮 Forecasting",
    "💾 Export"
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — RINGKASAN DATA
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Ringkasan Data")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observasi",  f"{len(df)} bulan")
    c2.metric("Periode",          f"{df['tahun'].min()} – {df['tahun'].max()}")
    c3.metric("Missing Values",   int(df[FEATURES].isnull().sum().sum()))
    c4.metric("Jumlah Tahun",     df['tahun'].nunique())

    st.subheader("Statistik Deskriptif (juta Rp)")
    desc = df[FEATURES].describe().round(2)
    desc.index = ['Jumlah','Rata-rata','Std Dev','Min','Q1','Median','Q3','Maks']
    st.dataframe(desc, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Koefisien Variasi (%)")
        cv = {f: round((df[f].std()/df[f].mean())*100, 1) for f in FEATURES}
        st.dataframe(pd.DataFrame(cv, index=['CV (%)']).T.rename_axis("Variabel"),
                     use_container_width=True)
    with cb:
        st.subheader("Uji Normalitas Shapiro-Wilk (α=0.05)")
        rows = []
        for f in FEATURES:
            stat, p = stats.shapiro(df[f])
            rows.append({'Variabel': f, 'Statistik': round(stat,4),
                         'p-value': round(p,4),
                         'Distribusi': 'Normal' if p > 0.05 else 'Tidak Berdistribusi Normal'})
        st.dataframe(pd.DataFrame(rows).set_index('Variabel'), use_container_width=True)
        st.caption("→ Data tidak normal → normalisasi Z-Score tepat digunakan sebelum clustering.")

    st.subheader("Rasio Arus Kas / Pendapatan per Tahun (%)")
    rasio = df.groupby('tahun').apply(
        lambda g: round(g['arus_kas_operasi'].sum() / g['pendapatan_operasi'].sum() * 100, 2)
    ).reset_index(name='Rasio (%)')
    rasio['Keterangan'] = rasio['Rasio (%)'].apply(lambda r: '⚠️ ANOMALI' if r < 2 else '✅ Normal')
    st.dataframe(rasio.set_index('tahun'), use_container_width=True)

    st.subheader("Preview Data")
    st.dataframe(df_raw, use_container_width=True, height=300)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — PENENTUAN K OPTIMAL
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Penentuan K Optimal")

    scaler_opt = StandardScaler()
    Xs_opt     = scaler_opt.fit_transform(df_raw[FEATURES].values)

    K_range, inertia, sil_scores, db_scores = range(2,9), [], [], []
    with st.spinner("Menghitung metrik untuk tiap K..."):
        for k in K_range:
            km     = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(Xs_opt)
            inertia.append(km.inertia_)
            sil_scores.append(silhouette_score(Xs_opt, labels))
            db_scores.append(davies_bouldin_score(Xs_opt, labels))

    # Deteksi elbow
    ia   = np.array(inertia); kv = np.array(list(K_range))
    p1, p2 = np.array([kv[0], ia[0]]), np.array([kv[-1], ia[-1]])
    dists  = [np.abs(np.cross(p2-p1, p1-np.array([kv[i], ia[i]]))) / np.linalg.norm(p2-p1)
              for i in range(len(kv))]
    elbow_k    = int(kv[np.argmax(dists)])
    sil_best_k = list(K_range)[sil_scores.index(max(sil_scores))]

    metrics_df = pd.DataFrame({
        'K': list(K_range),
        'Inertia': [round(v,2) for v in inertia],
        'Silhouette Score': [round(v,4) for v in sil_scores],
        'Davies-Bouldin': [round(v,4) for v in db_scores]
    }).set_index('K')
    st.dataframe(metrics_df, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Elbow terdeteksi",    f"K = {elbow_k}")
    c2.metric("Silhouette tertinggi", f"K = {sil_best_k} ({max(sil_scores):.4f})")
    c3.metric("K yang digunakan",    f"K = {K}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('#FAFAFA')
    axes[0].plot(list(K_range), inertia, 'o-', color='#2980b9', linewidth=2.5, markersize=8)
    axes[0].axvline(x=K, color='#e74c3c', linestyle='--', linewidth=2, label=f'K terpilih = {K}')
    axes[0].set_title('Elbow Method (Inertia)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Jumlah Klaster (K)'); axes[0].set_ylabel('Inertia')
    axes[0].legend(fontsize=10); axes[0].grid(alpha=0.3); axes[0].set_facecolor('#FAFAFA')

    axes[1].plot(list(K_range), sil_scores, 's-', color='#8e44ad', linewidth=2.5, markersize=8)
    axes[1].axvline(x=K, color='#e74c3c', linestyle='--', linewidth=2, label=f'K terpilih = {K}')
    axes[1].set_title('Silhouette Score', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Jumlah Klaster (K)'); axes[1].set_ylabel('Silhouette Score')
    axes[1].legend(fontsize=10); axes[1].grid(alpha=0.3); axes[1].set_facecolor('#FAFAFA')
    plt.tight_layout()
    st.pyplot(fig)

    with st.expander("📝 Interpretasi pemilihan K"):
        idx_K    = list(K_range).index(K)
        selisih  = max(sil_scores) - sil_scores[idx_K]
        st.markdown(f"""
- **Elbow Method** menunjukkan siku paling jelas pada **K = {elbow_k}**.
- Silhouette Score tertinggi ada di **K = {sil_best_k}** ({max(sil_scores):.4f}), namun K = {K} ({sil_scores[idx_K]:.4f}) tidak berbeda jauh (selisih {selisih:.4f}).
- **K = {K}** dipilih karena menghasilkan {K} klaster yang bisa diinterpretasikan secara manajerial: {", ".join(NAMA_K)}.
        """)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — HASIL CLUSTERING
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader(f"Hasil K-Means Clustering (K={K})")

    c1, c2 = st.columns(2)
    c1.metric("Silhouette Score",    f"{sil:.4f}", help="Mendekati 1 = sangat baik")
    c2.metric("Davies-Bouldin Index", f"{db:.4f}",  help="Mendekati 0 = sangat baik")

    # Warna per klaster di tabel (pakai highlight)
    st.subheader("Statistik Per Klaster (rata-rata, juta Rp)")
    summary = df.groupby('klaster')[FEATURES].mean().round(1)
    summary.columns = ['Rata-rata Arus Kas', 'Rata-rata Pendapatan', 'Rata-rata Beban']
    summary = summary.reindex([n for n in NAMA_K if n in summary.index])
    st.dataframe(summary, use_container_width=True)

    ca, cb = st.columns(2)
    with ca:
        st.subheader("Jumlah Bulan Per Klaster")
        cnt = df['klaster'].value_counts().reindex(aktif).rename('Jumlah Bulan').to_frame()
        st.dataframe(cnt, use_container_width=True)
    with cb:
        st.subheader("Distribusi Klaster per Tahun")
        cross = df.groupby(['tahun','klaster']).size().unstack(fill_value=0)
        cross = cross.reindex(columns=[n for n in NAMA_K if n in cross.columns])
        st.dataframe(cross, use_container_width=True)

    if 2023 in df['tahun'].values:
        st.subheader("Analisis Anomali 2023")
        df23   = df[df['tahun']==2023]
        r23    = df23['arus_kas_operasi'].mean() / df23['pendapatan_operasi'].mean() * 100
        r_else = df[df['tahun']!=2023]['arus_kas_operasi'].mean() / \
                 df[df['tahun']!=2023]['pendapatan_operasi'].mean() * 100
        c1, c2 = st.columns(2)
        c1.metric("Rasio Arus Kas/Pendapatan 2023",     f"{r23:.2f}%")
        c2.metric("Rasio Arus Kas/Pendapatan Non-2023",  f"{r_else:.2f}%")
        if r23 < 2:
            st.warning("⚠️ Tahun 2023 menunjukkan anomali: pendapatan tertinggi namun arus kas operasi sangat rendah.")

    st.subheader("Data Hasil Clustering")
    st.dataframe(df[['tahun','bulan','arus_kas_operasi','pendapatan_operasi',
                      'beban_operasi','klaster']], use_container_width=True, height=350)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — VISUALISASI
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Visualisasi Hasil Clustering")

    df_s = df.sort_values(['tahun','bulan_num']).reset_index(drop=True)

    # ── Grafik 1: Tren arus kas bulanan
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
    ax1.set_xticks([]); ax1.legend(fontsize=8, loc='upper left', ncol=min(K,3))
    ax1.grid(axis='y', alpha=0.25); ax1.set_facecolor('white')
    plt.tight_layout(); st.pyplot(fig1)

    c1, c2 = st.columns(2)

    # ── Grafik 2: Scatter pendapatan vs arus kas
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
        ax2.legend(fontsize=7); ax2.grid(alpha=0.25); ax2.set_facecolor('white')
        plt.tight_layout(); st.pyplot(fig2)

    # ── Grafik 3: Distribusi klaster per tahun
    with c2:
        st.markdown("#### Distribusi Klaster per Tahun")
        fig3, ax3 = plt.subplots(figsize=(6,4))
        fig3.patch.set_facecolor('#FAFAFA')
        cross_t = df.groupby(['tahun','klaster']).size().unstack(fill_value=0)
        cols_ok = [n for n in NAMA_K if n in cross_t.columns]
        cross_t[cols_ok].plot(kind='bar', ax=ax3,
                              color=[COLORS[c] for c in cols_ok],
                              edgecolor='white', width=0.7)
        ax3.set_xlabel('Tahun', fontsize=9); ax3.set_ylabel('Jumlah Bulan', fontsize=9)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax3.legend(fontsize=7); ax3.grid(axis='y', alpha=0.25); ax3.set_facecolor('white')
        plt.tight_layout(); st.pyplot(fig3)

    # ── Grafik 4: Heatmap
    st.markdown("#### Heatmap Klaster Per Bulan Per Tahun")
    klaster_num_map  = {kl: i for i, kl in enumerate(aktif)}
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
    # Inisial otomatis: huruf pertama tiap kata
    inisial_map = {kl: ''.join(w[0] for w in kl.split()) for kl in aktif}
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                label = inisial_map.get(aktif[int(val)], '?')
                ax4.text(j, i, label, ha='center', va='center',
                         fontsize=9, fontweight='bold', color='white')
    plt.tight_layout(); st.pyplot(fig4)


# ═══════════════════════════════════════════════════════════════
# TAB 5 — FORECASTING (Linear Regression, periode dinamis)
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("🔮 Forecasting Kinerja Keuangan")
    st.caption("Metode: Linear Regression | Fitur: indeks waktu bulanan")

    VAR_LABELS = {
        'arus_kas_operasi':   'Arus Kas Operasi',
        'pendapatan_operasi': 'Pendapatan Operasi',
        'beban_operasi':      'Beban Operasi',
    }
    VAR_COLORS = {
        'arus_kas_operasi':   '#2980b9',
        'pendapatan_operasi': '#1a9e3f',
        'beban_operasi':      '#e74c3c',
    }

    # ── Kontrol periode forecast ──────────────────────────────
    col_ctrl1, col_ctrl2 = st.columns([2, 1])
    with col_ctrl1:
        FORECAST_N = st.slider(
            "Jumlah bulan yang ingin di-forecast",
            min_value=1, max_value=60, value=12, step=1,
            help="Maksimal 60 bulan (5 tahun ke depan)"
        )
    with col_ctrl2:
        tahun_ke   = FORECAST_N / 12
        st.metric("Periode forecast", f"{FORECAST_N} bulan",
                  delta=f"≈ {tahun_ke:.1f} tahun" if FORECAST_N >= 12 else None)

    # ── Siapkan data time-series ──────────────────────────────
    df_ts = df.sort_values(['tahun','bulan_num']).reset_index(drop=True)
    df_ts['t'] = np.arange(len(df_ts))

    hist_labels = [f"{row['bulan']} {row['tahun']}" for _, row in df_ts.iterrows()]

    last_tahun = int(df_ts['tahun'].iloc[-1])
    last_bulan = int(df_ts['bulan_num'].iloc[-1])
    future_labels = []
    for i in range(1, FORECAST_N + 1):
        bnum  = (last_bulan - 1 + i) % 12
        tahun = last_tahun + (last_bulan - 1 + i) // 12
        future_labels.append(f"{BULAN_ORDER[bnum]} {tahun}")

    # Info periode
    st.info(f"📅 Forecast dari **{future_labels[0]}** sampai **{future_labels[-1]}**")

    # ── Latih model & forecast per variabel ──────────────────
    results = {}
    for var in FEATURES:
        X_train     = df_ts[['t']].values
        y_train     = df_ts[var].values
        model       = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_hist = model.predict(X_train)
        t_future    = np.arange(len(df_ts), len(df_ts) + FORECAST_N).reshape(-1, 1)
        y_forecast  = model.predict(t_future)
        results[var] = {
            'y_hist':      y_train,
            'y_pred_hist': y_pred_hist,
            'y_forecast':  y_forecast,
            'mae':  mean_absolute_error(y_train, y_pred_hist),
            'rmse': np.sqrt(mean_squared_error(y_train, y_pred_hist)),
            'r2':   r2_score(y_train, y_pred_hist),
        }

    # ── Metrik akurasi ───────────────────────────────────────
    st.divider()
    st.markdown("#### Evaluasi Model (data historis)")
    met_cols = st.columns(3)
    for i, var in enumerate(FEATURES):
        r = results[var]
        with met_cols[i]:
            st.markdown(f"**{VAR_LABELS[var]}**")
            # Warna R² — hijau jika >= 0.5, kuning jika >= 0.3, merah jika < 0.3
            st.metric("R²",   f"{r['r2']:.4f}",  delta=r2_delta, delta_color="off",
                      help="Mendekati 1 = fit sangat baik")
            st.metric("MAE",  f"{r['mae']:,.0f}", help="Rata-rata error absolut (juta Rp)")
            st.metric("RMSE", f"{r['rmse']:,.0f}",help="Root mean squared error (juta Rp)")

    with st.expander("ℹ️ Cara membaca metrik ini"):
        st.markdown("""
- **R²** — seberapa baik garis tren cocok dengan data historis. Nilai ≥ 0.7 sangat baik, 0.5–0.7 cukup, < 0.3 artinya data terlalu fluktuatif untuk tren linear.
- **MAE** — rata-rata selisih antara nilai aktual dan prediksi (juta Rp). Makin kecil makin akurat.
- **RMSE** — mirip MAE tapi lebih sensitif terhadap error yang besar.
- R² arus kas yang rendah bukan berarti model salah — bisa jadi memang arus kas sangat fluktuatif. Hal ini bisa dijadikan temuan di laporan PKL.
        """)

    st.divider()

    # ── Grafik per variabel ───────────────────────────────────
    st.markdown("#### Grafik Forecast per Variabel")

    # Tentukan jarak tick X tergantung total panjang data
    n_hist   = len(df_ts)
    n_total  = n_hist + FORECAST_N
    tick_step = 6 if n_total <= 120 else 12

    for var in FEATURES:
        r     = results[var]
        color = VAR_COLORS[var]
        label = VAR_LABELS[var]

        x_hist = list(range(n_hist))
        x_fore = list(range(n_hist, n_hist + FORECAST_N))

        fig, ax = plt.subplots(figsize=(14, 3.8))
        fig.patch.set_facecolor('#FAFAFA')

        # Aktual
        ax.plot(x_hist, r['y_hist'], color=color, linewidth=1.4,
                alpha=0.55, label='Aktual', zorder=2)
        ax.scatter(x_hist, r['y_hist'], color=color, s=20, zorder=3,
                   edgecolors='white', linewidth=0.4)

        # Tren fit
        ax.plot(x_hist, r['y_pred_hist'], color='#555', linewidth=1.1,
                linestyle='--', alpha=0.65, label='Tren (fit)', zorder=2)

        # Forecast line
        ax.plot([x_hist[-1]] + x_fore,
                [r['y_hist'][-1]] + list(r['y_forecast']),
                color=color, linewidth=2, linestyle='-',
                label=f'Forecast ({FORECAST_N} bln)', zorder=3)

        # Titik forecast — tampilkan jika <= 24 bulan, skip jika lebih
        if FORECAST_N <= 24:
            ax.scatter(x_fore, r['y_forecast'], color=color, s=45, zorder=4,
                       edgecolors='white', linewidth=0.7, marker='D')

        # Area forecast
        ax.axvspan(n_hist - 0.5, n_hist + FORECAST_N - 0.5,
                   alpha=0.07, color=color)
        ax.axvline(x=n_hist - 0.5, color='#888', linewidth=1,
                   linestyle=':', alpha=0.8)

        # Label "Forecast →"
        ymax = max(r['y_hist'].max(), r['y_forecast'].max())
        ymin = min(r['y_hist'].min(), r['y_forecast'].min())
        ax.text(n_hist + 0.3, ymax - (ymax - ymin) * 0.05,
                'Forecast →', fontsize=8, color='#888', va='top')

        # Anotasi nilai akhir forecast
        ax.annotate(f"Rp {r['y_forecast'][-1]:,.0f}",
                    xy=(x_fore[-1], r['y_forecast'][-1]),
                    xytext=(x_fore[-1] - max(3, FORECAST_N // 5),
                            r['y_forecast'][-1] + r['y_hist'].std() * 0.45),
                    fontsize=8, color=color, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.1))

        # X ticks — adaptif
        all_labels = hist_labels + future_labels
        tick_hist  = list(range(0, n_hist, tick_step))
        # Untuk forecast: tampilkan setiap tick_step bulan
        tick_fore  = list(range(n_hist, n_hist + FORECAST_N, max(1, FORECAST_N // 6)))
        if x_fore[-1] not in tick_fore:
            tick_fore.append(x_fore[-1])
        tick_pos  = tick_hist + tick_fore
        tick_labs = [all_labels[i] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labs, rotation=45, ha='right', fontsize=7.5)

        ax.set_title(f'{label} — Aktual & Forecast {FORECAST_N} Bulan ke Depan',
                     fontsize=11, fontweight='bold')
        ax.set_ylabel('Juta Rp', fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:,.0f}'))
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.25)
        ax.set_facecolor('white')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Tabel hasil forecast ─────────────────────────────────
    st.divider()
    st.markdown(f"#### Tabel Hasil Forecast {FORECAST_N} Bulan ({future_labels[0]} – {future_labels[-1]})")
    forecast_df = pd.DataFrame({
        'Periode':               future_labels,
        'Arus Kas (juta Rp)':   [round(v) for v in results['arus_kas_operasi']['y_forecast']],
        'Pendapatan (juta Rp)': [round(v) for v in results['pendapatan_operasi']['y_forecast']],
        'Beban (juta Rp)':      [round(v) for v in results['beban_operasi']['y_forecast']],
    })
    forecast_df['Est. Laba Operasi (juta Rp)'] = (
        forecast_df['Pendapatan (juta Rp)'] - forecast_df['Beban (juta Rp)']
    )
    st.dataframe(forecast_df.set_index('Periode'), use_container_width=True)
    st.info("💡 **Est. Laba Operasi** = Pendapatan − Beban hasil forecast.")

    # Simpan ke session state untuk export
    st.session_state['forecast_df']    = forecast_df
    st.session_state['forecast_label'] = f'Forecast {FORECAST_N} Bulan'


# ─────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Export Hasil")

    out_cols = ['tahun','bulan','arus_kas_operasi','pendapatan_operasi','beban_operasi','klaster']
    out_df   = df[out_cols].copy()
    st.dataframe(out_df, use_container_width=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        out_df.to_excel(writer, index=False, sheet_name='Hasil Clustering')
        if 'forecast_df' in st.session_state:
            sheet_label = st.session_state.get('forecast_label', 'Forecast')[:31]
            st.session_state['forecast_df'].to_excel(
                writer, index=False, sheet_name=sheet_label
            )
    buf.seek(0)

    st.download_button(
        label="⬇️ Download hasil_clustering_forecast.xlsx",
        data=buf,
        file_name="hasil_clustering_forecast.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    st.info("File Excel berisi 2 sheet: **Hasil Clustering** dan **Forecast 12 Bulan** (buka tab Forecasting dulu agar sheet forecast tersedia).")
