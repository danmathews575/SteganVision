# Evaluation Package
from .metrics import MetricsCalculator, compute_cohens_d, paired_t_test, bonferroni_correction
from .steganalyzer import Steganalyzer, SteganalysisDataset, SteganalyzerTrainer, evaluate_steganalyzer
from .frequency_analysis import analyze_batch_spectra, plot_spectral_comparison
