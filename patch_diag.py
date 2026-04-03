with open('create_diagnostic_notebook.py', 'r', encoding='utf-8') as f:
    code = f.read()

code = code.replace(
    "X_xgb = np.hstack([X_latent, regime_probs, X_base_selected])",
    "X_xgb = np.hstack([X_latent, X_base_selected, regime_probs])"
)

code = code.replace(
    """[f'regime_{i}' for i in range(num_states)] + \\
                        [feature_cols[i] for i in top_indices]""",
    """[feature_cols[i] for i in top_indices] + \\
                        [f'regime_{i}' for i in range(num_states)]"""
)

code = code.replace(
    """[f'regime_{i}' for i in range(num_states)] + \\
                        feature_cols""",
    """feature_cols + \\
                        [f'regime_{i}' for i in range(num_states)]"""
)

with open('create_diagnostic_notebook.py', 'w', encoding='utf-8') as f:
    f.write(code)
