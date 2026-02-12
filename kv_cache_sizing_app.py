"""
KV Cache Sizing Calculator - Streamlit App

Calculate cache capacity, sustainable RPS, and achievable hit rates
for LLM serving with KV cache reuse.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from model_loader import model_loader


# ============================================================================
# Helper Functions for Calculations
# ============================================================================

def calculate_rps_random(S_bytes, T, b, H_sec, r, h):
    """
    Calculate max sustainable RPS using random eviction policy.
    Formula: λ = (S × r) / (T × b × H × h)
    """
    if h == 0:
        return 0
    return (S_bytes * r) / (T * b * H_sec * h)


def calculate_rps_oracle(S_bytes, T, b, H_sec, h):
    """
    Calculate max sustainable RPS using oracle eviction policy.
    Formula: λ = S / (T × b × H × h)
    """
    if h == 0:
        return 0
    return S_bytes / (T * b * H_sec * h)


def calculate_hit_rate_random(S_bytes, lam, T, b, H_sec, r):
    """
    Calculate achievable hit rate using random eviction policy.
    Formula: h = r × min(1, S / (λ × T × b × H))
    """
    if lam == 0:
        return r
    return r * min(1.0, S_bytes / (lam * T * b * H_sec))


def calculate_hit_rate_oracle(S_bytes, lam, T, b, H_sec, r):
    """
    Calculate achievable hit rate using oracle eviction policy.
    Formula: h = min(r, S / (λ × T × b × H))
    """
    if lam == 0:
        return r
    return min(r, S_bytes / (lam * T * b * H_sec))


def calculate_capacity_random(h, lam, T, b, H_sec, r):
    """
    Calculate required capacity using random eviction policy.
    Formula: S = (h/r) × λ × T × b × H
    """
    if r == 0:
        return 0
    return (h / r) * lam * T * b * H_sec


def calculate_capacity_oracle(h, lam, T, b, H_sec):
    """
    Calculate required capacity using oracle eviction policy.
    Formula: S = h × λ × T × b × H
    """
    return h * lam * T * b * H_sec


def bytes_to_human(bytes_val):
    """Convert bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB', 'PB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def calculate_concurrency(rps, ttft_sec, output_tokens, tbt_sec):
    """Calculate expected concurrency using Little's Law."""
    time_in_system = ttft_sec + (output_tokens * tbt_sec)
    return rps * time_in_system


# ============================================================================
# Streamlit App Configuration
# ============================================================================

st.set_page_config(
    page_title="KV Cache Sizing Calculator",
    page_icon="📊",
    layout="wide",
)

st.title("🧮 KV Cache Sizing Calculator")
st.markdown("""
Calculate cache capacity, sustainable RPS, and achievable hit rates for LLM serving with KV cache reuse.
Compares **Random** eviction (lower bound) vs **Oracle** eviction (upper bound) policies.
""")

# ============================================================================
# Create Tabs
# ============================================================================

tab1, tab2, tab3 = st.tabs([
    "📈 View 1: Capacity → RPS",
    "🎯 View 2: Workload → Hit Rate",
    "💾 View 3: Target → Capacity"
])

# ============================================================================
# VIEW 1: Capacity → Sustainable RPS
# ============================================================================

with tab1:
    st.header("View 1: Capacity → Sustainable RPS")
    st.markdown("Given available cache capacity, calculate the maximum sustainable request rate.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")

        # Model selection
        model_names = model_loader.list_models()
        selected_model = st.selectbox(
            "LLM Model",
            model_names,
            index=model_names.index('meta-llama--Meta-Llama-3.1-8B') if 'meta-llama--Meta-Llama-3.1-8B' in model_names else 0,
            key="v1_model"
        )

        # Data type
        dtype = st.selectbox("Data Type", ["FP16", "BF16", "FP8", "FP32"], index=0, key="v1_dtype")

        # Get KV bytes per token
        model_config = model_loader.get_model(selected_model)
        kv_bytes = model_config.get_kv_bytes_per_token(dtype)

        # Capacity
        capacity_gb = st.number_input(
            "Available Capacity (GB)",
            min_value=0.1,
            max_value=10000000.0,
            value=10.0,
            step=1.0,
            key="v1_capacity"
        )
        capacity_bytes = capacity_gb * (1024 ** 3)

        # Sequence parameters
        st.markdown("**Sequence Parameters**")
        input_tokens = st.number_input("Input Tokens (Context)", min_value=1, value=10000, step=1000, key="v1_input")
        output_tokens = st.number_input("Output Tokens", min_value=1, value=1000, step=100, key="v1_output")
        total_tokens = input_tokens + output_tokens

        # Cache parameters
        st.markdown("**Cache Parameters**")
        reuse_fraction = st.slider(
            "Workload Reuse Fraction (r)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Fraction of tokens that will be reused within the horizon",
            key="v1_r"
        )

        target_hit_rate = st.slider(
            "Target Hit Rate (h)",
            min_value=0.01,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Target fraction of cache lookups that hit",
            key="v1_h"
        )

        reuse_horizon_hours = st.number_input(
            "Reuse Horizon (hours)",
            min_value=0.1,
            max_value=48.0,
            value=1.0,
            step=0.5,
            help="Time window for reuse (e.g., p95 reuse gap)",
            key="v1_horizon"
        )
        reuse_horizon_sec = reuse_horizon_hours * 3600

        # SLO parameters
        st.markdown("**SLO Parameters**")
        slo_ttft = st.number_input(
            "SLO TTFT (seconds)",
            min_value=0.01,
            max_value=10.0,
            value=0.5,
            step=0.1,
            key="v1_ttft"
        )

        slo_tbt = st.number_input(
            "SLO TBT (milliseconds)",
            min_value=1.0,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            key="v1_tbt"
        )
        slo_tbt_sec = slo_tbt / 1000

    with col2:
        st.subheader("Results")

        # Calculate RPS
        rps_random = calculate_rps_random(
            capacity_bytes, total_tokens, kv_bytes, reuse_horizon_sec, reuse_fraction, target_hit_rate
        )
        rps_oracle = calculate_rps_oracle(
            capacity_bytes, total_tokens, kv_bytes, reuse_horizon_sec, target_hit_rate
        )

        # Calculate concurrency
        conc_random = calculate_concurrency(rps_random, slo_ttft, output_tokens, slo_tbt_sec)
        conc_oracle = calculate_concurrency(rps_oracle, slo_ttft, output_tokens, slo_tbt_sec)

        # Display results
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.metric("KV Bytes per Token", f"{kv_bytes:,} bytes ({kv_bytes/1024:.1f} KB)")
            st.metric("Total Tokens per Request", f"{total_tokens:,}")
            st.metric("Working Set Size", bytes_to_human(rps_oracle * total_tokens * kv_bytes * reuse_horizon_sec))

        with col_r2:
            st.metric("Time in System", f"{slo_ttft + output_tokens * slo_tbt_sec:.2f} sec")
            st.metric("Capacity Utilization (Oracle)", f"{(rps_oracle * total_tokens * kv_bytes * reuse_horizon_sec / capacity_bytes * 100):.1f}%")

        st.markdown("---")

        # RPS Results
        st.markdown("### Maximum Sustainable RPS")
        col_rps1, col_rps2 = st.columns(2)

        with col_rps1:
            st.markdown("**Random Eviction Policy** (Lower Bound)")
            st.metric("Max RPS", f"{rps_random:.2f}")
            st.metric("Concurrency", f"{conc_random:.1f} requests")

        with col_rps2:
            st.markdown("**Oracle Eviction Policy** (Upper Bound)")
            st.metric("Max RPS", f"{rps_oracle:.2f}")
            st.metric("Concurrency", f"{conc_oracle:.1f} requests")

        # Plot: RPS vs Hit Rate
        st.markdown("### RPS vs Hit Rate")

        h_values = np.linspace(0.01, min(1.0, reuse_fraction), 100)
        rps_random_curve = [calculate_rps_random(capacity_bytes, total_tokens, kv_bytes, reuse_horizon_sec, reuse_fraction, h) for h in h_values]
        rps_oracle_curve = [calculate_rps_oracle(capacity_bytes, total_tokens, kv_bytes, reuse_horizon_sec, h) for h in h_values]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(h_values, rps_oracle_curve, 'b-', linewidth=2.5, label='Oracle (Upper Bound)')
        ax.plot(h_values, rps_random_curve, 'r--', linewidth=2.0, label='Random (Lower Bound)')
        ax.axvline(target_hit_rate, color='gray', linestyle=':', alpha=0.6, label=f'Target h={target_hit_rate:.2f}')
        ax.fill_between(h_values, rps_random_curve, rps_oracle_curve, alpha=0.2, color='gray')

        ax.set_xlabel("Hit Rate (h)", fontsize=11)
        ax.set_ylabel("Max Sustainable RPS", fontsize=11)
        ax.set_title(f"Capacity={capacity_gb:.1f} GB, T={total_tokens:,}, r={reuse_fraction:.2f}, H={reuse_horizon_hours:.1f}h", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        st.pyplot(fig)

        # CSV Export
        st.markdown("### Export Results")

        # Prepare summary data
        summary_data = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Model': [selected_model],
            'Data_Type': [dtype],
            'Capacity_GB': [capacity_gb],
            'Input_Tokens': [input_tokens],
            'Output_Tokens': [output_tokens],
            'Total_Tokens': [total_tokens],
            'Reuse_Fraction': [reuse_fraction],
            'Target_Hit_Rate': [target_hit_rate],
            'Reuse_Horizon_Hours': [reuse_horizon_hours],
            'SLO_TTFT_Sec': [slo_ttft],
            'SLO_TBT_Ms': [slo_tbt],
            'KV_Bytes_Per_Token': [kv_bytes],
            'Max_RPS_Random': [rps_random],
            'Max_RPS_Oracle': [rps_oracle],
            'Concurrency_Random': [conc_random],
            'Concurrency_Oracle': [conc_oracle],
            'Time_In_System_Sec': [slo_ttft + output_tokens * slo_tbt_sec],
            'Working_Set_Bytes': [rps_oracle * total_tokens * kv_bytes * reuse_horizon_sec],
        }

        # Prepare curve data
        curve_data = {
            'Hit_Rate': h_values,
            'RPS_Random': rps_random_curve,
            'RPS_Oracle': rps_oracle_curve,
        }

        df_summary = pd.DataFrame(summary_data)
        df_curve = pd.DataFrame(curve_data)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_summary = df_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_summary,
                file_name=f"view1_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with col_exp2:
            csv_curve = df_curve.to_csv(index=False)
            st.download_button(
                label="📥 Download Curve Data CSV",
                data=csv_curve,
                file_name=f"view1_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )


# ============================================================================
# VIEW 2: Workload → Achievable Hit Rate
# ============================================================================

with tab2:
    st.header("View 2: Workload → Achievable Hit Rate")
    st.markdown("Given workload characteristics, calculate the achievable cache hit rate.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")

        # Model selection
        selected_model_v2 = st.selectbox(
            "LLM Model",
            model_names,
            index=model_names.index('meta-llama--Meta-Llama-3.1-70B') if 'meta-llama--Meta-Llama-3.1-70B' in model_names else 0,
            key="v2_model"
        )

        dtype_v2 = st.selectbox("Data Type", ["FP16", "BF16", "FP8", "FP32"], index=2, key="v2_dtype")

        model_config_v2 = model_loader.get_model(selected_model_v2)
        kv_bytes_v2 = model_config_v2.get_kv_bytes_per_token(dtype_v2)


        # Capacity
        capacity_gb_v2 = st.number_input(
            "Available Capacity (GB)",
            min_value=0.1,
            max_value=10000.0,
            value=100.0,
            step=10.0,
            key="v2_capacity"
        )
        capacity_bytes_v2 = capacity_gb_v2 * (1024 ** 3)

        # Workload
        st.markdown("**Workload Parameters**")
        request_rate = st.number_input(
            "Request Rate (RPS)",
            min_value=0.1,
            max_value=1000.0,
            value=7.0,
            step=1.0,
            key="v2_rps"
        )

        seq_length_v2 = st.number_input(
            "Sequence Length (tokens)",
            min_value=1,
            value=10000,
            step=1000,
            key="v2_seq"
        )

        reuse_fraction_v2 = st.slider(
            "Workload Reuse Fraction (r)",
            min_value=0.0,
            max_value=1.0,
            value=0.99,
            step=0.01,
            key="v2_r"
        )

        reuse_horizon_hours_v2 = st.number_input(
            "Reuse Horizon (hours)",
            min_value=0.1,
            max_value=48.0,
            value=1.0,
            step=0.5,
            key="v2_horizon"
        )
        reuse_horizon_sec_v2 = reuse_horizon_hours_v2 * 3600

    with col2:
        st.subheader("Results")

        # Calculate hit rates
        hit_rate_random = calculate_hit_rate_random(
            capacity_bytes_v2, request_rate, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2
        )
        hit_rate_oracle = calculate_hit_rate_oracle(
            capacity_bytes_v2, request_rate, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2
        )

        working_set = request_rate * seq_length_v2 * kv_bytes_v2 * reuse_horizon_sec_v2

        # Display results
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.metric("KV Bytes per Token", f"{kv_bytes_v2:,} bytes ({kv_bytes_v2/1024:.1f} KB)")
            st.metric("Working Set (W)", bytes_to_human(working_set))

        with col_r2:
            st.metric("Capacity / Working Set", f"{capacity_bytes_v2 / working_set:.2f}x")
            st.metric("Theoretical Max Hit Rate", f"{reuse_fraction_v2:.2%}")

        st.markdown("---")

        # Hit rate results
        st.markdown("### Achievable Hit Rates")
        col_hr1, col_hr2 = st.columns(2)

        with col_hr1:
            st.markdown("**Random Eviction Policy** (Lower Bound)")
            st.metric("Hit Rate", f"{hit_rate_random:.2%}")
            #st.metric("% of Theoretical Max", f"{(hit_rate_random / reuse_fraction_v2 * 100):.1f}%")

        with col_hr2:
            st.markdown("**Oracle Eviction Policy** (Upper Bound)")
            st.metric("Hit Rate", f"{hit_rate_oracle:.2%}")
            #st.metric("% of Theoretical Max", f"{(hit_rate_oracle / reuse_fraction_v2 * 100):.1f}%")

        # Plot: Hit Rate vs Capacity
        st.markdown("### Hit Rate vs Capacity")

        # Sweep capacity
        S_values = np.logspace(9, 15, 200)  # 1 GB to ~1 PB
        rps_list = [1, 3, request_rate, 10] if request_rate not in [1, 3, 10] else [1, 3, 7, 10]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(rps_list)))

        for c, lam in zip(colors, rps_list):
            h_rand = [calculate_hit_rate_random(S, lam, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2) for S in S_values]
            h_oracle = [calculate_hit_rate_oracle(S, lam, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2) for S in S_values]

            linestyle = '-' if lam == request_rate else '--'
            linewidth = 2.5 if lam == request_rate else 1.5
            alpha = 1.0 if lam == request_rate else 0.6

            ax.semilogx(S_values / (1024**3), h_rand, color=c, linestyle=linestyle, linewidth=linewidth, alpha=alpha, label=f"λ={lam} RPS (random)")
            ax.semilogx(S_values / (1024**3), h_oracle, color=c, linestyle=':', linewidth=linewidth, alpha=alpha, label=f"λ={lam} RPS (oracle)")

        ax.axvline(capacity_gb_v2, color='red', linestyle=':', linewidth=2, alpha=0.7, label=f'Current Capacity')

        ax.set_xlabel("Cache Capacity (GB)", fontsize=11)
        ax.set_ylabel("Hit Rate", fontsize=11)
        ax.set_title(f"T={seq_length_v2:,}, r={reuse_fraction_v2:.2f}, H={reuse_horizon_hours_v2:.1f}h", fontsize=12)
        ax.set_ylim(0, min(1.05, reuse_fraction_v2 + 0.05))
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        st.pyplot(fig)

        # CSV Export
        st.markdown("### Export Results")

        # Prepare summary data
        summary_data_v2 = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Model': [selected_model_v2],
            'Data_Type': [dtype_v2],
            'Capacity_GB': [capacity_gb_v2],
            'Request_Rate_RPS': [request_rate],
            'Sequence_Length': [seq_length_v2],
            'Reuse_Fraction': [reuse_fraction_v2],
            'Reuse_Horizon_Hours': [reuse_horizon_hours_v2],
            'KV_Bytes_Per_Token': [kv_bytes_v2],
            'Hit_Rate_Random': [hit_rate_random],
            'Hit_Rate_Oracle': [hit_rate_oracle],
            'Working_Set_Bytes': [working_set],
            'Capacity_Over_WorkingSet': [capacity_bytes_v2 / working_set],
            'Pct_of_Max_Random': [hit_rate_random / reuse_fraction_v2 * 100],
            'Pct_of_Max_Oracle': [hit_rate_oracle / reuse_fraction_v2 * 100],
        }

        # Prepare curve data for all RPS values
        curve_data_v2 = {'Capacity_GB': S_values / (1024**3)}
        for lam in rps_list:
            h_rand = [calculate_hit_rate_random(S, lam, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2) for S in S_values]
            h_oracle = [calculate_hit_rate_oracle(S, lam, seq_length_v2, kv_bytes_v2, reuse_horizon_sec_v2, reuse_fraction_v2) for S in S_values]
            curve_data_v2[f'Hit_Rate_Random_RPS_{lam}'] = h_rand
            curve_data_v2[f'Hit_Rate_Oracle_RPS_{lam}'] = h_oracle

        df_summary_v2 = pd.DataFrame(summary_data_v2)
        df_curve_v2 = pd.DataFrame(curve_data_v2)

        col_exp1, col_exp2 = st.columns(2)
        with col_exp1:
            csv_summary_v2 = df_summary_v2.to_csv(index=False)
            st.download_button(
                label="📥 Download Summary CSV",
                data=csv_summary_v2,
                file_name=f"view2_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="v2_summary"
            )

        with col_exp2:
            csv_curve_v2 = df_curve_v2.to_csv(index=False)
            st.download_button(
                label="📥 Download Curve Data CSV",
                data=csv_curve_v2,
                file_name=f"view2_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="v2_curve"
            )


# ============================================================================
# VIEW 3: Target RPS & Hit Rate → Required Capacity
# ============================================================================

with tab3:
    st.header("View 3: Target RPS & Hit Rate → Required Capacity")
    st.markdown("Given target RPS and desired hit rate, calculate the required cache capacity.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Inputs")

        # Model selection
        selected_model_v3 = st.selectbox(
            "LLM Model",
            model_names,
            index=model_names.index('meta-llama--Meta-Llama-3.1-70B') if 'meta-llama--Meta-Llama-3.1-70B' in model_names else 0,
            key="v3_model"
        )

        dtype_v3 = st.selectbox("Data Type", ["FP16", "BF16", "FP8", "FP32"], index=2, key="v3_dtype")

        model_config_v3 = model_loader.get_model(selected_model_v3)
        kv_bytes_v3 = model_config_v3.get_kv_bytes_per_token(dtype_v3)

        # Target parameters
        st.markdown("**Target Parameters**")
        target_rps = st.number_input(
            "Target RPS",
            min_value=0.1,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            key="v3_rps"
        )

        target_hit_v3 = st.slider(
            "Target Hit Rate",
            min_value=0.01,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="v3_hit"
        )

        # Workload parameters
        st.markdown("**Workload Parameters**")
        seq_length_v3 = st.number_input(
            "Sequence Length (tokens)",
            min_value=1,
            value=10000,
            step=1000,
            key="v3_seq"
        )

        reuse_fraction_v3 = st.slider(
            "Workload Reuse Fraction (r)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            key="v3_r"
        )

        reuse_horizon_hours_v3 = st.number_input(
            "Reuse Horizon (hours)",
            min_value=0.1,
            max_value=48.0,
            value=1.0,
            step=0.5,
            key="v3_horizon"
        )
        reuse_horizon_sec_v3 = reuse_horizon_hours_v3 * 3600

    with col2:
        st.subheader("Results")

        # Calculate required capacity
        capacity_random = calculate_capacity_random(
            target_hit_v3, target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3
        )
        capacity_oracle = calculate_capacity_oracle(
            target_hit_v3, target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3
        )

        # Display results
        col_r1, col_r2 = st.columns(2)

        with col_r1:
            st.metric("KV Bytes per Token", f"{kv_bytes_v3:,} bytes ({kv_bytes_v3/1024:.1f} KB)")
            st.metric("Working Set (W)", bytes_to_human(target_rps * seq_length_v3 * kv_bytes_v3 * reuse_horizon_sec_v3))

        with col_r2:
            st.metric("Theoretical Max Hit Rate", f"{reuse_fraction_v3:.2%}")
            if target_hit_v3 > reuse_fraction_v3:
                st.warning(f"⚠️ Target hit rate ({target_hit_v3:.2%}) exceeds reuse fraction ({reuse_fraction_v3:.2%})")

        st.markdown("---")

        # Capacity results
        st.markdown("### Required Cache Capacity")
        col_cap1, col_cap2 = st.columns(2)

        with col_cap1:
            st.markdown("**Random Eviction Policy** (Conservative)")
            st.metric("Required Capacity", bytes_to_human(capacity_random))
            st.metric("Capacity (GB)", f"{capacity_random / (1024**3):.2f} GB")

        with col_cap2:
            st.markdown("**Oracle Eviction Policy** (Optimistic)")
            st.metric("Required Capacity", bytes_to_human(capacity_oracle))
            st.metric("Capacity (GB)", f"{capacity_oracle / (1024**3):.2f} GB")

        st.info(f"💡 **Capacity Range**: {capacity_oracle / (1024**3):.1f} GB (oracle) to {capacity_random / (1024**3):.1f} GB (random)")

        # Plot: Capacity vs Hit Rate for fixed RPS
        st.markdown("### Capacity vs Hit Rate")

        h_sweep = np.linspace(0.01, min(1.0, reuse_fraction_v3), 100)
        cap_random_curve = [calculate_capacity_random(h, target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3) / (1024**3) for h in h_sweep]
        cap_oracle_curve = [calculate_capacity_oracle(h, target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3) / (1024**3) for h in h_sweep]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(h_sweep, cap_oracle_curve, 'b-', linewidth=2.5, label='Oracle (Optimistic)')
        ax.plot(h_sweep, cap_random_curve, 'r--', linewidth=2.0, label='Random (Conservative)')
        ax.axvline(target_hit_v3, color='green', linestyle=':', linewidth=2, alpha=0.7, label=f'Target h={target_hit_v3:.2f}')
        ax.axhline(capacity_random / (1024**3), color='red', linestyle=':', alpha=0.5)
        ax.axhline(capacity_oracle / (1024**3), color='blue', linestyle=':', alpha=0.5)
        ax.fill_between(h_sweep, cap_random_curve, cap_oracle_curve, alpha=0.2, color='gray')

        ax.set_xlabel("Hit Rate (h)", fontsize=11)
        ax.set_ylabel("Required Capacity (GB)", fontsize=11)
        ax.set_title(f"λ={target_rps:.1f} RPS, T={seq_length_v3:,}, r={reuse_fraction_v3:.2f}, H={reuse_horizon_hours_v3:.1f}h", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        st.pyplot(fig)

        # Reference lines for common capacities
        st.markdown("### Reference Capacity Tiers")
        tiers = [
            ("HBM (8x GPU)", 1.5 * 1024),  # 1.5 TB
            ("HBM + DDR", 5.5 * 1024),     # 5.5 TB
            ("HBM + DDR + NVMe", 35.5 * 1024),  # 35.5 TB
        ]

        tier_data = []
        for tier_name, tier_gb in tiers:
            achievable_hit_random = calculate_hit_rate_random(
                tier_gb * (1024**3), target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3
            )
            achievable_hit_oracle = calculate_hit_rate_oracle(
                tier_gb * (1024**3), target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3
            )
            tier_data.append({
                "Tier": tier_name,
                "Capacity": f"{tier_gb:.1f} GB",
                "Hit Rate (Random)": f"{achievable_hit_random:.2%}",
                "Hit Rate (Oracle)": f"{achievable_hit_oracle:.2%}",
            })

        st.table(tier_data)

        # CSV Export
        st.markdown("### Export Results")

        # Prepare summary data
        summary_data_v3 = {
            'Timestamp': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            'Model': [selected_model_v3],
            'Data_Type': [dtype_v3],
            'Target_RPS': [target_rps],
            'Target_Hit_Rate': [target_hit_v3],
            'Sequence_Length': [seq_length_v3],
            'Reuse_Fraction': [reuse_fraction_v3],
            'Reuse_Horizon_Hours': [reuse_horizon_hours_v3],
            'KV_Bytes_Per_Token': [kv_bytes_v3],
            'Required_Capacity_GB_Random': [capacity_random / (1024**3)],
            'Required_Capacity_GB_Oracle': [capacity_oracle / (1024**3)],
            'Required_Capacity_Bytes_Random': [capacity_random],
            'Required_Capacity_Bytes_Oracle': [capacity_oracle],
            'Working_Set_Bytes': [target_rps * seq_length_v3 * kv_bytes_v3 * reuse_horizon_sec_v3],
        }

        # Prepare curve data
        curve_data_v3 = {
            'Hit_Rate': h_sweep,
            'Capacity_GB_Random': cap_random_curve,
            'Capacity_GB_Oracle': cap_oracle_curve,
        }

        # Prepare tier data for export
        tier_export_data = []
        for tier_name, tier_gb in tiers:
            achievable_hit_random = calculate_hit_rate_random(
                tier_gb * (1024**3), target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3
            )
            achievable_hit_oracle = calculate_hit_rate_oracle(
                tier_gb * (1024**3), target_rps, seq_length_v3, kv_bytes_v3, reuse_horizon_sec_v3, reuse_fraction_v3
            )
            tier_export_data.append({
                "Tier": tier_name,
                "Capacity_GB": tier_gb,
                "Hit_Rate_Random": achievable_hit_random,
                "Hit_Rate_Oracle": achievable_hit_oracle,
            })

        df_summary_v3 = pd.DataFrame(summary_data_v3)
        df_curve_v3 = pd.DataFrame(curve_data_v3)
        df_tiers = pd.DataFrame(tier_export_data)

        col_exp1, col_exp2, col_exp3 = st.columns(3)
        with col_exp1:
            csv_summary_v3 = df_summary_v3.to_csv(index=False)
            st.download_button(
                label="📥 Summary CSV",
                data=csv_summary_v3,
                file_name=f"view3_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="v3_summary"
            )

        with col_exp2:
            csv_curve_v3 = df_curve_v3.to_csv(index=False)
            st.download_button(
                label="📥 Curve Data CSV",
                data=csv_curve_v3,
                file_name=f"view3_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="v3_curve"
            )

        with col_exp3:
            csv_tiers = df_tiers.to_csv(index=False)
            st.download_button(
                label="📥 Tiers CSV",
                data=csv_tiers,
                file_name=f"view3_tiers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="v3_tiers"
            )


# ============================================================================
# Footer
# ============================================================================

st.markdown("---")
st.markdown("""
**Formulas:**
- Random Policy: `λ = (S × r) / (T × b × H × h)`, `h = r × min(1, S / (λ × T × b × H))`
- Oracle Policy: `λ = S / (T × b × H × h)`, `h = min(r, S / (λ × T × b × H))`

Where:
- S = Cache capacity (bytes)
- λ = Request rate (RPS)
- T = Sequence length (tokens)
- b = KV bytes per token
- H = Reuse horizon (seconds)
- r = Reuse fraction (0-1)
- h = Hit rate (0-1)
""")
