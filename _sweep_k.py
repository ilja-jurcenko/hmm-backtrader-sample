import datetime, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib.machinery import SourceFileLoader
_eval = SourceFileLoader('hmm_eval', os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'hmm-eval.py')).load_module()

csv_path = os.path.join('datas', 'spy-2000-2026.csv')
ts = datetime.datetime(2014, 1, 1)
te = datetime.datetime(2019, 1, 1)
test_e = datetime.datetime(2025, 1, 1)

best_feats = ['log_ret', 'r5', 'r20', 'vol_short', 'atr_norm']
k_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
pca_vals = [None, 2, 3]

hdr = f"{'K':>3} {'PCA':>4} {'Acc':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'BIC':>12}"
print(hdr)
print('-' * len(hdr))

best_f1, best_k = 0, ''
for K in k_vals:
    for pca in pca_vals:
        r = _eval.evaluate_once(
            ticker='SPY', csv_path=csv_path,
            train_start=ts, train_end=te,
            test_start=te, test_end=test_e,
            n_components=K, hmm_features=best_feats, hmm_pca=pca,
            gt_method='sma', gt_window=40,
            verbose=False,
        )
        if not r:
            continue
        pca_s = str(pca) if pca else '-'
        print(f"{K:>3} {pca_s:>4}  {r['accuracy']:.4f}  {r['precision']:.4f}"
              f"  {r['recall']:.4f}  {r['f1']:.4f} {r.get('bic', 0):>12.1f}")
        if r['f1'] > best_f1:
            best_f1 = r['f1']
            best_k = f'K={K} pca={pca}'

print(f"\nBest: {best_k}  F1={best_f1:.4f}")
