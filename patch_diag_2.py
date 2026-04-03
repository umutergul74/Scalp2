with open('create_diagnostic_notebook.py', 'r', encoding='utf-8') as f:
    code = f.read()

old_block = """                "shap_values = explainer.shap_values(X_sample_xgb)\\n",
                "\\n",
                "plt.figure(figsize=(10, 8))\\n",
                "plt.title(f'SHAP Summary - Fold {TARGET_FOLD}', fontsize=16)\\n",
                "shap.summary_plot(shap_values, X_sample_xgb, feature_names=feature_names_xgb, show=False)\\n","""

new_block = """                "shap_values = explainer.shap_values(X_sample_xgb)\\n",
                "\\n",
                "if isinstance(shap_values, list):\\n",
                "    shap_long = shap_values[2]\\n",
                "elif len(np.array(shap_values).shape) == 3:\\n",
                "    shap_long = shap_values[:, :, 2]\\n",
                "else:\\n",
                "    shap_long = shap_values\\n",
                "\\n",
                "plt.figure(figsize=(10, 8))\\n",
                "plt.title(f'SHAP Summary (Long Yönü) - Fold {TARGET_FOLD}', fontsize=16)\\n",
                "shap.summary_plot(shap_long, X_sample_xgb, feature_names=feature_names_xgb, show=False)\\n","""

code = code.replace(old_block, new_block)

with open('create_diagnostic_notebook.py', 'w', encoding='utf-8') as f:
    f.write(code)
