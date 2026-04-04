import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# 05.1 \u2014 Model Analytics & Deep Diagnostics\n",
                "Bu notebook, canl\u0131 da \u00e7al\u0131\u015fan modelin (Fold 053) derinlemesine istatistiksel MR'\u0131n\u0131 \u00e7eker.\n",
                "Klasik metriklerin \u00f6tesinde; IC (Information Coefficient), Kalibrasyon (Brier Score), latent uzay\u0131 t-SNE g\u00f6sterimi ve SHAP analizi yap\u0131larak sistemin a\u015f\u0131r\u0131 uyum (overfitting) durumu kontrol edilir."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Gerekli k\u00fct\u00fcphaneleri kural\u0131m\n",
                "!pip install -q xgboost shap scikit-learn matplotlib seaborn pandas numpy\n",
                "!pip install -q pyarrow fastparquet hmmlearn ccxt PyWavelets numba"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "drive.mount('/content/drive')\n",
                "\n",
                "import os, sys, json, pickle\n",
                "REPO_DIR = '/content/scalp2_repo'\n",
                "if os.path.exists(os.path.join(REPO_DIR, '.git')):\n",
                "    !git -C {REPO_DIR} pull --ff-only\n",
                "else:\n",
                "    !git clone https://github.com/sergul74/Scalp2.git {REPO_DIR}\n",
                "\n",
                "sys.path.insert(0, REPO_DIR)\n",
                "\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import torch\n",
                "from sklearn.preprocessing import RobustScaler\n",
                "\n",
                "from scalp2.config import load_config\n",
                "config = load_config(f'{REPO_DIR}/config.yaml')\n",
                "\n",
                "DATA_DIR = '/content/drive/MyDrive/scalp2/data/processed'\n",
                "CHECKPOINT_DIR = '/content/drive/MyDrive/scalp2/checkpoints'"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import shap\n",
                "import xgboost as xgb\n",
                "from scipy.stats import spearmanr\n",
                "from sklearn.calibration import calibration_curve\n",
                "from sklearn.metrics import brier_score_loss\n",
                "from sklearn.manifold import TSNE\n",
                "from sklearn.decomposition import PCA\n",
                "from scalp2.utils.serialization import load_fold_artifacts\n",
                "from scalp2.models.hybrid import HybridEncoder\n",
                "from scalp2.data.dataset import ScalpDataset\n",
                "from torch.utils.data import DataLoader\n",
                "\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams['figure.figsize'] = (10, 6)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Mimarinin (Fold 053) Y\u00fcklenmesi"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "TARGET_FOLD = 53\n",
                "print(f'Analiz Edilen Model: Fold {TARGET_FOLD}')\n",
                "\n",
                "df = pd.read_parquet(f'{DATA_DIR}/BTC_USDT_labeled.parquet')\n",
                "with open(f'{DATA_DIR}/feature_columns.json', 'r') as f:\n",
                "    feature_cols = json.load(f)\n",
                "\n",
                "features_array = df[feature_cols].values\n",
                "labels_array = df['tb_label_cls'].values\n",
                "returns_array = df['tb_return'].values\n",
                "\n",
                "artifacts = load_fold_artifacts(CHECKPOINT_DIR, TARGET_FOLD)\n",
                "\n",
                "n_features = len(feature_cols)\n",
                "model_tcn = HybridEncoder(n_features, config.model)\n",
                "model_tcn.load_state_dict(artifacts['model_state'])\n",
                "model_tcn.eval()\n",
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
                "model_tcn.to(device)\n",
                "\n",
                "scaler = artifacts['scaler']\n",
                "regime_detector = artifacts['regime_detector']\n",
                "top_indices = artifacts.get('top_feature_indices')\n",
                "\n",
                "print('Artifacts y\u00fcklendi. Model haz\u0131r.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Walk-Forward Test Verisinden Kalibrasyon / IC Score Analizi\n",
                "Wf_predictions i\u00e7erisindeki Test periyodu sonu\u00e7lar\u0131 kullan\u0131larak modelin \"ne derece d\u00fcr\u00fcst\" oldu\u011fu test edilir."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "try:\n",
                "    with open(f'{DATA_DIR}/wf_predictions.pkl', 'rb') as f:\n",
                "        wf_predictions = pickle.load(f)\n",
                "    \n",
                "    fold_data = next((f for f in wf_predictions if f['fold_idx'] == TARGET_FOLD), None)\n",
                "    \n",
                "    if fold_data:\n",
                "        probs = fold_data['test_probabilities']\n",
                "        labels = fold_data['test_labels']\n",
                "        test_returns = returns_array[fold_data['test_start'] + config.model.seq_len : fold_data['test_end']][:len(labels)]\n",
                "        \n",
                "        is_long = (labels == 2).astype(int)\n",
                "        b_score = brier_score_loss(is_long, probs[:, 2])\n",
                "        print(f'\\u25b6 Brier Score (Kalibrasyon Hatas\u0131 - Long Y\u00f6n\u00fc): {b_score:.4f} (D\u00fc\u015f\u00fck Olmal\u0131)')\n",
                "        \n",
                "        prob_true, prob_pred = calibration_curve(is_long, probs[:, 2], n_bins=10)\n",
                "        \n",
                "        plt.figure(figsize=(8,6))\n",
                "        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label='XGBoost Fold 053 (Long)')\n",
                "        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='M\u00fckemmel Olas\u0131l\u0131k (1:1)')\n",
                "        plt.title('Reliability Diagram (Kalibrasyon E\u011frisi) - OOS Test', fontsize=14)\n",
                "        plt.xlabel('Tahmin Edilen Long Olas\u0131l\u0131\u011f\u0131', fontsize=12)\n",
                "        plt.ylabel('Ger\u00e7ekle\u015fen Ba\u015far\u0131 (Long Hedefe Ula\u015fma)', fontsize=12)\n",
                "        plt.legend()\n",
                "        plt.savefig('diag_calibration.png', bbox_inches='tight')\n",
                "        plt.show()\n",
                "        \n",
                "        net_prob = probs[:, 2] - probs[:, 0]\n",
                "        rank_ic, p_val = spearmanr(net_prob, test_returns)\n",
                "        print(f'\\n\\u25b6 Rank IC (Information Coefficient - Net Y\u00f6n): {rank_ic:.4f} (P-Value: {p_val:.4f})')\n",
                "        if rank_ic > 0.03:\n",
                "            print('\\u2705 Model, k\u00e2rl\u0131 fiyat hareketi b\u00fcy\u00fckl\u00fc\u011f\u00fcn\u00fc de a\u011f\u0131rl\u0131kland\u0131rarak b\u00fcy\u00fck olas\u0131l\u0131k verebiliyor (Pozitif Korelasyon).\\n')\n",
                "        else:\n",
                "            print('\\u26a0\\ufe0f Zay\u0131f IC. Model yanl\u0131\u015f hedeflerde b\u00fcy\u00fck stop loss yiyor olabilir.\\n')\n",
                "            \n",
                "        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
                "        y_pred = np.argmax(probs, axis=1)\n",
                "        print(f'\\n\\u25b6 MULTICLASS METRICS:')\n",
                "        print(f'Accuracy: {accuracy_score(labels, y_pred):.4f}')\n",
                "        print(f'Precision (Long): {precision_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}')\n",
                "        print(f'Recall (Long): {recall_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}')\n",
                "        print(f'F1 Score (Long): {f1_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}\\n')\n",
                "        \n",
                "        with open('diag_metrics.txt', 'w') as mf:\n",
                "            mf.write(f'Brier Score: {b_score:.4f}\\n')\n",
                "            mf.write(f'Rank IC: {rank_ic:.4f} (p={p_val:.4f})\\n')\n",
                "            mf.write(f'Accuracy: {accuracy_score(labels, y_pred):.4f}\\n')\n",
                "            mf.write(f'Precision (Long): {precision_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}\\n')\n",
                "            mf.write(f'Recall (Long): {recall_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}\\n')\n",
                "            mf.write(f'F1 Score (Long): {f1_score(labels, y_pred, labels=[2], average=\"macro\", zero_division=0):.4f}\\n')\n",
                "            \n",
                "    else:\n",
                "        print(\"\\u26a0\\ufe0f Fold 053 i\u00e7in OOS prediction verisiwf_predictions.pkl i\u00e7inde bulunamad\u0131.\")\n",
                "except Exception as e:\n",
                "    print('wf_predictions okunamad\u0131:', e)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Deep Learning \"Latent Space\" Analizi (t-SNE & PCA)\n",
                "TCN+GRU modelinin i\u00e7 \u00f6znitelik \u00e7\u0131kar\u0131m yetene\u011fi incelenir. S\u0131n\u0131flar (Long vs Short/N\u00f6tr) \u00fcst \u00fcste karma\u015f\u0131ksa \u00f6\u011frenme de\u011fil ezberleme ger\u00e7ekle\u015fiyot demektir."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "sample_idx_end = -1\n",
                "sample_idx_start = -5000\n",
                "\n",
                "sample_df = df.iloc[sample_idx_start:sample_idx_end].copy()\n",
                "sample_feat = features_array[sample_idx_start:sample_idx_end]\n",
                "sample_labels = labels_array[sample_idx_start:sample_idx_end]\n",
                "\n",
                "scaled_sample = scaler.transform(sample_feat).astype(np.float32)\n",
                "dummy_returns = np.zeros_like(sample_labels, dtype=np.float32)\n",
                "dataset = ScalpDataset(scaled_sample, sample_labels, dummy_returns, config.model.seq_len)\n",
                "loader = DataLoader(dataset, batch_size=config.training.batch_size, shuffle=False)\n",
                "\n",
                "latents = []\n",
                "with torch.no_grad():\n",
                "    for x, _, _ in loader:\n",
                "        x = x.to(device)\n",
                "        _, latent = model_tcn(x)\n",
                "        latents.append(latent.cpu().numpy())\n",
                "\n",
                "X_latent = np.vstack(latents)\n",
                "y_latent_labels = sample_labels[config.model.seq_len:]\n",
                "\n",
                "regime_probs = regime_detector.predict_proba(sample_df.iloc[config.model.seq_len:])\n",
                "num_states = regime_probs.shape[1]\n",
                "\n",
                "X_base = scaled_sample[config.model.seq_len:]\n",
                "if top_indices is not None and len(top_indices) > 0:\n",
                "    X_base_selected = X_base[:, top_indices]\n",
                "    feature_names_xgb = [f'latent_{i}' for i in range(X_latent.shape[1])] + \\\n",
                "                        [f'regime_{i}' for i in range(num_states)] + \\\n",
                "                        [feature_cols[i] for i in top_indices]\n",
                "else:\n",
                "    X_base_selected = X_base\n",
                "    feature_names_xgb = [f'latent_{i}' for i in range(X_latent.shape[1])] + \\\n",
                "                        [f'regime_{i}' for i in range(num_states)] + \\\n",
                "                        feature_cols\n",
                "\n",
                "X_xgb = np.hstack([X_latent, X_base_selected, regime_probs])\n",
                "print(f\"Feature matrisi (\u00f6rnek uzay\u0131) \u00fcretildi: {X_xgb.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "pca = PCA(n_components=2)\n",
                "X_pca = pca.fit_transform(X_latent)\n",
                "print(f'\\u25b6 PCA \u0130zah Edilen Varyans Oran\u0131: {np.sum(pca.explained_variance_ratio_):.4f} (Ne kadar y\u00fcksekse, veriyi iki boyutta ifade etmek o kadar m\u00fcmk\u00fcn)')\n",
                "\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))\n",
                "scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_latent_labels, cmap='viridis', alpha=0.5, s=15)\n",
                "ax1.set_title('PCA İzdüşümü (Doğrusal Latent Ayrımlar)')\n",
                "ax1.set_xlabel('PCA 1')\n",
                "ax1.set_ylabel('PCA 2')\n",
                "\n",
                "print('t-SNE i\u015flemi y\u00fcr\u00fct\u00fcl\u00fcyor (Yakla\u015f\u0131k 10-20 sn)...')\n",
                "tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)\n",
                "X_tsne = tsne.fit_transform(X_latent)\n",
                "\n",
                "scatter2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_latent_labels, cmap='viridis', alpha=0.5, s=15)\n",
                "ax2.set_title('t-SNE İzdüşümü (Non-Linear Manifold)')\n",
                "ax2.set_xlabel('t-SNE 1')\n",
                "ax2.set_ylabel('t-SNE 2')\n",
                "\n",
                "plt.suptitle('Deep Learning (TCN+GRU) \u00d6znitelik Ayr\u0131\u015ft\u0131rma Performans\u0131', fontsize=15)\n",
                "plt.savefig('diag_latent.png', bbox_inches='tight')\n",
                "plt.show()\n",
                "with open('diag_metrics.txt', 'a') as mf:\n",
                "    mf.write(f'PCA Explained Var: {np.sum(pca.explained_variance_ratio_):.4f}\\n')\n",
                "print('\\n\\u2728 YORUM: Noktalar renklerine g\u00f6re adalara b\u00f6l\u00fcn\u00fc\u00fcyorsa model ayr\u0131\u015ft\u0131rma i\u015fini \u00e7\u00f6zm\u00fc\u015ft\u00fcr. Tam bir g\u00fcr\u00fclt\u00fc e\u011frisi (yuvarlak duman) ise DL k\u0131sm\u0131 zay\u0131f kalm\u0131\u015ft\u0131r.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. XGBoost Karar Föktörleri (SHAP Feature Importance)\n",
                "Nihai karar XGBoost tarafindan veriliyor. Hangi özellikler(Latent vs Raw vs Regime) ne yönde ağırlık katıyor teyitedelim."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "booster = xgb.Booster()\n",
                "xgb_path = f'{CHECKPOINT_DIR}/fold_{TARGET_FOLD:03d}/xgb_fold_{TARGET_FOLD:03d}.json'\n",
                "booster.load_model(xgb_path)\n",
                "\n",
                "print('SHAP De\u011ferleri Kaplan\u0131yor...')\n",
                "explainer = shap.TreeExplainer(booster)\n",
                "X_sample_xgb = X_xgb[:1500]  # Hiz icin ilk 1500 satir\n",
                "shap_values = explainer.shap_values(X_sample_xgb)\n",
                "\n",
                "if isinstance(shap_values, list):\n",
                "    shap_long = shap_values[2]\n",
                "elif len(np.array(shap_values).shape) == 3:\n",
                "    shap_long = shap_values[:, :, 2]\n",
                "else:\n",
                "    shap_long = shap_values\n",
                "\n",
                "plt.figure(figsize=(10, 8))\n",
                "plt.title(f'SHAP Summary (Long Yönü) - Fold {TARGET_FOLD}', fontsize=16)\n",
                "shap.summary_plot(shap_long, X_sample_xgb, feature_names=feature_names_xgb, show=False)\n",
                "plt.tight_layout()\n",
                "plt.savefig('diag_shap.png', bbox_inches='tight')\n",
                "plt.show()\n",
                "with open('diag_metrics.txt', 'a') as mf:\n",
                "    mf.write(f'SHAP Top 3 (Long): {feature_names_xgb[np.abs(shap_long).mean(0).argsort()[::-1][0]]}, {feature_names_xgb[np.abs(shap_long).mean(0).argsort()[::-1][1]]}, {feature_names_xgb[np.abs(shap_long).mean(0).argsort()[::-1][2]]}\\n')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Çıktıları Paketle (İndirmek İçin)\n",
                "Yukarıda oluşturulan tüm metrikleri ve PNG grafiklerini tek bir ZIP dosyasında toplar."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import shutil\n",
                "import os\n",
                "\n",
                "print('Çıktılar paketleniyor...')\n",
                "os.makedirs('diag_export', exist_ok=True)\n",
                "files_to_copy = ['diag_metrics.txt', 'diag_calibration.png', 'diag_latent.png', 'diag_shap.png']\n",
                "\n",
                "for f in files_to_copy:\n",
                "    if os.path.exists(f):\n",
                "        shutil.copy(f, f'diag_export/{f}')\n",
                "        print(f'{f} eklendi.')\n",
                "\n",
                "shutil.make_archive('diagnostic_results', 'zip', 'diag_export')\n",
                "print('\\n✅ BİTTİ! Colab sol menüsünden diagnostic_results.zip dosyasını indirebilirsiniz.')"
            ]
        }
    ],
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "gpuType": "T4",
            "provenance": []
        },
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('c:/Users/Umut/Documents/PlatformIO/Projects/Scalp2/notebooks', exist_ok=True)
out_path = 'c:/Users/Umut/Documents/PlatformIO/Projects/Scalp2/notebooks/05.1_model_diagnostics.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook created at {out_path}")
