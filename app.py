import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, to_hex
import matplotlib.colors as mcolors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Ringkasan Data",
    "🔍 Penentuan K Optimal",
    "🎯 Hasil Clustering",
    "📈 Visualisasi",
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
                         'Distribusi': 'Normal' if p > 0.05 else 'Tidak Normal'})
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
# TAB 5 — EXPORT
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.subheader("Export Hasil")

    out_cols = ['tahun','bulan','arus_kas_operasi','pendapatan_operasi','beban_operasi','klaster']
    out_df   = df[out_cols].copy()
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
    st.info("Untuk menyimpan grafik, klik kanan pada grafik di tab Visualisasi → Save image as.")