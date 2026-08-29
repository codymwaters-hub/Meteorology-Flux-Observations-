import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Dataset EDA & Insights", layout="wide")
st.title("🔬 Deep Exploratory Data Analysis & Insights")

# ---------------------------------------------------------
# 1. FIXED DATA LOAD & FEATURE SYNTHESIS
# ---------------------------------------------------------
# Direct path to the bundled repository dataset
DATA_FILE = "Site_information_basic.csv"

try:
    df = pd.read_csv(DATA_FILE)
except Exception as e:
    st.error(f"Error loading '{DATA_FILE}': {e}")
    st.stop()

# Auto-generate word counts for large descriptive text fields
text_cols = [c for c in df.select_dtypes(include=["object"]).columns if df[c].astype(str).str.len().mean() > 50]
for tc in text_cols:
    if f"{tc}_word_count" not in df.columns:
        df[f"{tc}_word_count"] = df[tc].astype(str).apply(lambda x: len(x.split()))

# ---------------------------------------------------------
# 2. AUTOMATIC PRIMARY KEY & IDENTIFIER EXCLUSION
# ---------------------------------------------------------
detected_id_cols = []
for c in df.columns:
    is_unique = (df[c].nunique() == len(df)) and (len(df) > 1)
    has_id_keyword = any(k in c.lower() for k in ["id", "ps number", "ps_number", "serial", "index", "code", "guid"])
    if is_unique or has_id_keyword:
        detected_id_cols.append(c)

st.sidebar.header("⚙️ Column Settings")
ignored_cols = st.sidebar.multiselect(
    "Primary Key / Identifier Columns (Excluded from Stats):",
    options=df.columns.tolist(),
    default=detected_id_cols,
    help="These columns will be excluded from statistical profiles, correlations, and frequency distributions."
)

analysis_df = df.drop(columns=ignored_cols)
numeric_cols = analysis_df.select_dtypes(include=["number"]).columns.tolist()
qualitative_cols = analysis_df.select_dtypes(include=["object", "category"]).columns.tolist()

st.caption(
    f"**Dataset Overview:** {df.shape[0]:,} rows | {df.shape[1]:,} columns "
    f"({len(numeric_cols)} numeric and {len(qualitative_cols)} qualitative variables evaluated; "
    f"{len(ignored_cols)} ID/key columns excluded from stats)."
)

# ---------------------------------------------------------
# 3. ADVANCED QUANTITATIVE STATISTICAL ANALYSIS
# ---------------------------------------------------------
st.header("📈 Advanced Quantitative Statistical Profile")

if numeric_cols:
    quant_metrics = []
    for col in numeric_cols:
        series = analysis_df[col].dropna()
        if len(series) == 0:
            continue
        
        q25, q50, q75 = series.quantile([0.25, 0.50, 0.75])
        iqr = q75 - q25
        lower_fence = q25 - 1.5 * iqr
        upper_fence = q75 + 1.5 * iqr
        
        outliers = series[(series < lower_fence) | (series > upper_fence)]
        outlier_count = len(outliers)
        mean_val = series.mean()
        std_val = series.std()
        cv = (std_val / mean_val) if mean_val != 0 else 0
        
        quant_metrics.append({
            "Column": col,
            "Count": len(series),
            "Mean": mean_val,
            "Std Dev": std_val,
            "CV": cv,
            "Min": series.min(),
            "Q1 (25%)": q25,
            "Median / Q2 (50%)": q50,
            "Q3 (75%)": q75,
            "Max": series.max(),
            "IQR": iqr,
            "Lower Fence (Q1 - 1.5×IQR)": lower_fence,
            "Upper Fence (Q3 + 1.5×IQR)": upper_fence,
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),
            "Outlier Count": outlier_count,
            "Outlier (%)": f"{(outlier_count / len(series) * 100):.1f}%"
        })
    
    quant_df = pd.DataFrame(quant_metrics).set_index("Column")
    
    st.subheader("Parametric, Percentile & Outlier Threshold Metrics")
    st.dataframe(quant_df.style.format({
        "Mean": "{:,.2f}", 
        "Std Dev": "{:,.2f}", 
        "CV": "{:.3f}",
        "Min": "{:,.2f}", 
        "Q1 (25%)": "{:,.2f}", 
        "Median / Q2 (50%)": "{:,.2f}", 
        "Q3 (75%)": "{:,.2f}", 
        "Max": "{:,.2f}",
        "IQR": "{:,.2f}", 
        "Lower Fence (Q1 - 1.5×IQR)": "{:,.2f}", 
        "Upper Fence (Q3 + 1.5×IQR)": "{:,.2f}", 
        "Skewness": "{:.2f}", 
        "Kurtosis": "{:.2f}"
    }), use_container_width=True)

    # Correlation Suite
    st.subheader("🔗 Correlation Analysis")
    if len(numeric_cols) >= 2:
        corr_method = st.radio("Correlation Metric:", ["Pearson (Linear)", "Spearman (Rank / Non-linear)"], horizontal=True)
        method_key = "pearson" if "Pearson" in corr_method else "spearman"
        corr_mat = analysis_df[numeric_cols].corr(method=method_key)
        
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Correlation Matrix ({corr_method.split()[0]}):**")
            st.dataframe(
                corr_mat.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1).format("{:.3f}"),
                use_container_width=True
            )
        with c2:
            st.markdown("**Top Correlated Pairs:**")
            unstacked = corr_mat.unstack()
            pairs = unstacked[unstacked.index.get_level_values(0) < unstacked.index.get_level_values(1)].reset_index()
            pairs.columns = ["Variable A", "Variable B", "Correlation"]
            pairs["Abs_Corr"] = pairs["Correlation"].abs()
            pairs = pairs.sort_values("Abs_Corr", ascending=False).drop(columns=["Abs_Corr"])
            st.dataframe(pairs.style.format({"Correlation": "{:.3f}"}), use_container_width=True, hide_index=True)
    else:
        st.info("At least 2 non-key numeric columns are required to generate correlation matrices.")
else:
    st.info("No non-key numerical columns available for statistical profiling.")

# ---------------------------------------------------------
# 4. ADVANCED QUALITATIVE STATISTICAL ANALYSIS
# ---------------------------------------------------------
st.header("📋 Advanced Qualitative Statistical Insights")

if qualitative_cols:
    qual_metrics = []
    for col in qualitative_cols:
        series = analysis_df[col].astype(str)
        n_total = len(series)
        n_unique = analysis_df[col].nunique(dropna=True)
        missing_count = analysis_df[col].isnull().sum()
        
        mode_val = analysis_df[col].mode()[0] if not analysis_df[col].mode().empty else "N/A"
        mode_count = (analysis_df[col] == mode_val).sum() if mode_val != "N/A" else 0
        dominance_ratio = (mode_count / n_total) if n_total > 0 else 0
        
        probs = analysis_df[col].value_counts(normalize=True, dropna=True)
        entropy = -np.sum(probs * np.log2(probs + 1e-9)) if len(probs) > 1 else 0
        max_entropy = np.log2(len(probs)) if len(probs) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        qual_metrics.append({
            "Column": col,
            "Total Rows": n_total,
            "Unique Classes": n_unique,
            "Cardinality Ratio": f"{(n_unique / n_total):.3f}",
            "Top Mode (Most Frequent)": str(mode_val)[:30],
            "Mode Share (%)": f"{(dominance_ratio * 100):.1f}%",
            "Missing Rate (%)": f"{(missing_count / n_total * 100):.1f}%",
            "Diversity Index (Entropy)": round(entropy, 2),
            "Uniformity Score (0-1)": round(normalized_entropy, 2)
        })
    
    qual_summary_df = pd.DataFrame(qual_metrics).set_index("Column")
    st.subheader("Categorical Information & Diversity Metrics")
    st.dataframe(qual_summary_df, use_container_width=True)

    st.subheader("Categorical Value Distributions")
    cat_columns = [c for c in qualitative_cols if analysis_df[c].nunique() <= 30]
    
    if cat_columns:
        selected_cat = st.selectbox("Select Categorical Feature to Inspect:", cat_columns)
        b1, b2 = st.columns([2, 3])
        with b1:
            val_counts = analysis_df[selected_cat].value_counts(dropna=False).reset_index()
            val_counts.columns = [selected_cat, "Frequency"]
            val_counts["Percentage"] = (val_counts["Frequency"] / len(analysis_df) * 100).round(2).astype(str) + "%"
            st.dataframe(val_counts, use_container_width=True, hide_index=True)
        with b2:
            st.bar_chart(analysis_df[selected_cat].value_counts())
    else:
        st.info("No categorical columns with under 30 unique categories for distribution charting.")

# ---------------------------------------------------------
# 5. INTERACTIVE DATA TABLE, INSIGHTS & EXPORT
# ---------------------------------------------------------
st.header("✏️ Interactive Workspace & Export")
edited_df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

st.subheader("📝 Analyst Notes & Hypotheses")
insights = st.text_area("Record your statistical takeaways, anomalies, and hypotheses:", placeholder="Document your analytical insights here...")

col1, col2 = st.columns(2)
with col1:
    if st.button("💾 Overwrite Dataset"):
        edited_df.to_csv(DATA_FILE, index=False)
        st.success(f"'{DATA_FILE}' successfully updated!")

with col2:
    csv_bytes = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cleaned Dataset (CSV)",
        data=csv_bytes,
        file_name="defects_data_cleaned.csv",
        mime="text/csv"
    )
