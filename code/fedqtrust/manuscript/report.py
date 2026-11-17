"""Report only completed, explicitly identified jobs; no substitute experiments."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, pearsonr
from scipy.optimize import curve_fit
from .engine import save_json

def build_report(root):
    root = Path(root)
    tables = root / 'tables'
    figures = root / 'figures'
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    final = []
    rounds = []
    clients = []
    timing = []
    trust_final = []
    traces = []
    completed = []
    manifest = json.loads((root / 'plan.json').read_text())
    expected = {j['id'] for j in manifest['jobs']}
    for path in sorted((root / 'jobs').glob('*')):
        if not (path / 'DONE').exists() or path.name not in expected:
            continue
        cfg = json.loads((path / 'config.json').read_text())
        cfg.pop('datasets', None)
        cfg = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}
        cfg['job_id'] = path.name
        for filename, target in [
            ('final.csv', final),
            ('rounds.csv', rounds),
            ('clients.csv', clients),
            ('timing.csv', timing)
        ]:
            df = pd.read_csv(path / filename)
            for key, value in cfg.items():
                df['planned_' + key if key in df.columns else key] = value
            target.append(df)
        if (path / 'trust_trace.csv').exists():
            df = pd.read_csv(path / 'trust_trace.csv')
            for key, value in cfg.items():
                df['planned_' + key if key in df.columns else key] = value
            traces.append(df)
        # Manuscript reporting box: results/experiment_name/seed_X/ (small files only).
        layout = root / 'results' / cfg['experiment'] / f"seed_{cfg['seed']}" / path.name
        layout.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'final.csv', 'rounds.csv', 'clients.csv', 'timing.csv',
                     'final_trust.csv', 'trust_trace.csv', 'selections.json', 'metadata.json'):
            if (path / name).exists():
                (layout / name).write_bytes((path / name).read_bytes())
        if (path / 'final_trust.csv').exists():
            df = pd.read_csv(path / 'final_trust.csv')
            for key, value in cfg.items():
                df['planned_' + key if key in df.columns else key] = value
            trust_final.append(df)
        completed.append(path.name)
    coverage = pd.DataFrame([dict(
        job_id=j['id'],
        experiment=j['experiment'],
        status='COMPLETE' if j['id'] in completed else 'MISSING'
    ) for j in manifest['jobs']])
    coverage.to_csv(tables / 'coverage.csv', index=False)
    if not final:
        save_json(
            root / 'audit.json',
            dict(all_jobs_complete=False, completed=0, expected=len(expected), publication_ready=False)
        )
        return
    f = pd.concat(final, ignore_index=True)
    r = pd.concat(rounds, ignore_index=True)
    c = pd.concat(clients, ignore_index=True)
    t = pd.concat(timing, ignore_index=True)
    for name, df in [('final_results', f), ('round_metrics', r), ('client_metrics', c), ('timing', t)]:
        df.to_csv(tables / f'{name}.csv', index=False)
    # Keep experiment and treatment settings separate; aggregate seeds only.
    dims = [
        'experiment',
        'method',
        'attack',
        'fraction',
        'dataset',
        'clients',
        'k',
        'rounds',
        'epochs',
        'batch',
        'lr',
        'label_fraction',
        'sigma',
        'lie_z',
        'alpha',
        'beta',
        'gamma',
        'theta',
        'lambda_',
        'mu',
        'p',
        'reputation_decay',
        'window',
        'solver',
        'noise',
        'infrastructure'
    ]
    summary = f.groupby(
        dims,
        dropna=False
    )[['accuracy', 'weighted_f1', 'auc_roc']].agg(['mean', 'std', 'count']).reset_index()
    summary.columns = ['_'.join(filter(None, x)) if isinstance(x, tuple) else x for x in summary.columns]
    summary.to_csv(tables / 'results_mean_std.csv', index=False)
    # Macro average across modalities within each seed, never treat tasks as seed replicates.
    macro = f.groupby(
        [d for d in dims if d != 'dataset'] + ['seed'],
        dropna=False
    )[['accuracy', 'weighted_f1']].mean().reset_index()
    macro.to_csv(tables / 'macro_per_seed.csv', index=False)
    stats = []
    stat_dims = [d for d in dims if d not in ('method', 'dataset')]
    for key, g in macro.groupby(stat_dims, dropna=False):
        pivot = g.pivot(index='seed', columns='method', values='accuracy')
        if 'FedQTrust' not in pivot:
            continue
        for method in pivot.columns.drop('FedQTrust'):
            paired = pivot[['FedQTrust', method]].dropna()
            n = len(paired)
            diff = paired['FedQTrust'] - paired[method]
            p = 1.0 if n >= 2 and np.allclose(
                diff,
                0
            ) else float(wilcoxon(diff, method='exact').pvalue) if n >= 2 else np.nan
            stats.append(dict(zip(
                stat_dims,
                key
            )) | dict(comparison=method, n_seeds=n, mean_difference=float(diff.mean()), p_value=p, p_bonferroni=min(1.0, p * 42) if np.isfinite(p) else np.nan, complete_five_seeds=n == 5))
    pd.DataFrame(stats).to_csv(tables / 'paired_seed_statistics.csv', index=False)
    fits = []
    for jid, g in r[r.experiment == 'E3'].groupby('job_id'):
        trace = g.groupby('round').gradient_proxy.mean()
        x = trace.index.to_numpy()
        y = trace.to_numpy()
        if len(x) < 3:
            continue
        pars, _ = curve_fit(lambda a, b, d: b / np.sqrt(a) + d, x, y)
        pred = pars[0] / np.sqrt(x) + pars[1]
        den = float(((y - y.mean()) ** 2).sum())
        acc = g.groupby('round').accuracy.mean()
        hit = acc[acc >= 0.8]
        fits.append(dict(
            job_id=jid,
            a=pars[0],
            b=pars[1],
            r2=1 - float(((y - pred) ** 2).sum()) / den if den else np.nan,
            rounds=len(x),
            rounds_to_80=int(hit.index[0]) if len(hit) else np.nan
        ))
    pd.DataFrame(fits).to_csv(tables / 'convergence_fits.csv', index=False)
    detection = []
    for jid, g in c[c.attack_active & c.method.isin([
        'FedQTrust',
        'Krum',
        'FLTrust',
        'No-QUBO',
        'No-Trust',
        'No-Blockchain',
        'No-PQC',
        'No-Quantum'
    ])].groupby('job_id'):
        truth = g.is_malicious.astype(bool)
        pred = g.predicted_malicious.astype(bool)
        tp = int((truth & pred).sum())
        fp = int((~truth & pred).sum())
        pos = int(truth.sum())
        neg = int((~truth).sum())
        corr = pearsonr(
            g.trust,
            truth.astype(int)
        ) if g.trust.nunique() > 1 and truth.nunique() > 1 else (np.nan, np.nan)
        detection.append(dict(
            job_id=jid,
            decision_rule=g.decision_rule.iloc[0],
            tpr=tp / pos if pos else np.nan,
            fpr=fp / neg if neg else np.nan,
            malicious_observations=pos,
            honest_observations=neg,
            trust_malicious_correlation=float(corr[0])
        ))
    pd.DataFrame(detection).to_csv(tables / 'detection.csv', index=False)
    # Manuscript metric: predict malicious iff tau_i^(T) < theta over all K clients;
    # Pearson rho between tau_i^(T) and ground truth; poisoning and free-riding separately.
    spec_detection = []
    if trust_final:
        tf = pd.concat(trust_final, ignore_index=True)
        tf.to_csv(tables / 'final_trust.csv', index=False)
        for jid, g in tf[tf.attack != 'benign'].groupby('job_id'):
            for role in ('poisoner', 'free_rider'):
                part = g[g.role.isin([role, 'honest'])]
                truth = (part.role == role)
                if not truth.any():
                    continue
                pred = part.predicted_malicious.astype(bool)
                corr = pearsonr(part.trust, truth.astype(int)) if part.trust.nunique() > 1 else (np.nan, np.nan)
                spec_detection.append(dict(
                    job_id=jid, experiment=g.experiment.iloc[0], method=g.method.iloc[0],
                    attack=g.attack.iloc[0], fraction=g.fraction.iloc[0], seed=g.seed.iloc[0],
                    attack_class='free_riding' if role == 'free_rider' else 'poisoning',
                    tpr=float((truth & pred).sum() / truth.sum()),
                    fpr=float((~truth & pred).sum() / max((~truth).sum(), 1)),
                    pearson_rho=float(corr[0]), pearson_p=float(corr[1])))
    pd.DataFrame(spec_detection).to_csv(tables / 'detection_final_trust.csv', index=False)
    gain = f[f.experiment == 'E4'].pivot_table(
        index=['dataset', 'seed', 'attack'],
        columns='method',
        values='accuracy'
    )
    if {'FedQTrust', 'PQS-BFL'}.issubset(gain.columns):
        gain['difference'] = gain.FedQTrust - gain['PQS-BFL']
        gain.to_csv(tables / 'modality_gain.csv')
        # Dataset size from observed test metrics is not training size: obtain partition counts.
        sizes = {}
        for path in (root / 'partitions').glob('*.json'):
            partition = json.loads(path.read_text())
            sizes[partition['dataset']] = sum((len(x['train']) + len(x['validation']) for x in partition['clients'])) + len(partition['root_indices'])
        benign = gain.reset_index().query("attack == 'benign'").groupby('dataset').difference.mean()
        if len(benign) >= 3 and all((d in sizes for d in benign.index)):
            result = pearsonr(np.log10([sizes[d] for d in benign.index]), benign.values)
            save_json(
                tables / 'modality_size_correlation.json',
                dict(
                    r=float(result.statistic),
                    p=float(result.pvalue),
                    n_datasets=len(benign),
                    size_definition='official training split'
                )
            )
    if traces:
        pd.concat(traces, ignore_index=True).to_csv(tables / 'trust_trace.csv', index=False)
    _plots(figures, f, r, c, t, root)
    benchmark_complete = all(((root / 'benchmarks' / e / 'benchmark_status.json').exists() and (json.loads((root / 'benchmarks' / e / 'benchmark_status.json').read_text()).get(
        'complete',
        False
    ) and not json.loads((root / 'benchmarks' / e / 'benchmark_status.json').read_text()).get('quick', True)) for e in ('E2', 'E7') if e in manifest['experiments']))
    # Explicit unresolved scientific/protocol decisions are never hidden by a DONE marker.
    limitations = [
        'Exact server deployment and full paper grid require execution.',
        'Classical/QNN heads have about 0.5M parameters, not the manuscript 8K claim.',
        'ZZFeatureMap uses feature/angle encoding, not amplitude encoding.',
        'Sparse classes retain global label dimensions; labels are never changed per client.',
        'Krum f is bounded by available participants; requested Byzantine settings can violate its guarantees.',
        'LIE uses the largest z in [0, z_max] accepted by Krum (grid + binary search); if Krum accepts none, the Baruch et al. quantile z (capped at z_max) is used.',
        'Malicious client ids {1..floor(rK)} interleave modalities (id c -> dataset c mod 5), so every modality contains attackers.',
        'Label-flipping clients relabel a fixed seeded set of samples (label_fraction=1.0), so r equals the relabelled fraction of training samples.',
        'QAOA reports the most probable bitstring of the optimized circuit; best-sampled and expected objectives are logged separately. QAOA timeouts fall back to SA and are recorded per round.',
        'Secure transport is not privacy-preserving secure aggregation against the server.',
        'Fabric deployment, endorsement/consensus policy and on-chain signature integration require independent validation.',
        'PQS-BFL (ML-DSA-65 + ledger + CNN) and SecEdge-MC (ML-KEM + ML-DSA + ledger + CNN) follow Table 4 and have no Byzantine defence, so their accuracy equals FedAvg; they differ only in cryptographic/ledger overhead. They are protocol adaptations, not reproductions of external code.',
        'CITADEL is named in the abstract but not specified in the baseline table and is not run.',
        'QUBO pairwise term penalises dissimilar selected pairs, so identical colluding updates are not penalised against each other (Eq. 1 as specified).',
        'Nairobi-derived noise is an explicit homogeneous extrapolation beyond seven physical qubits.',
        'Krum pointwise rejection means not selected; its ROC uses the actual Krum distance score.',
        'Five seeds cannot achieve the proposed Bonferroni threshold with an exact two-sided signed-rank test.'
    ]
    local_ledger = bool((f.infrastructure == 'local').any())
    if local_ledger:
        limitations.insert(0, 'LOCAL LEDGER RUN: cryptography and training are measured; Fabric commit latency, consensus and on-chain verification are unmeasured. The No-Blockchain comparison cannot establish a Fabric benefit.')
    save_json(
        root / 'audit.json',
        dict(
            all_jobs_complete=expected == set(completed),
            benchmarks_complete=benchmark_complete,
            execution_scope='local_ledger_with_real_crypto' if local_ledger else 'configured_infrastructure',
            fabric_network_measured=bool((f.infrastructure == 'paper').all()),
            completed=len(completed),
            expected=len(expected),
            publication_ready=False,
            limitations=limitations
        )
    )
    _latex(root, summary, limitations)
    paper_tables(root, macro, f, t, tables)
    experiment_tables(root, f, r, t, macro)
    checklist = output_checklist(root)
    audit = json.loads((root / 'audit.json').read_text())
    audit['missing_outputs'] = [row['name'] for row in checklist if not row['present']]
    save_json(root / 'audit.json', audit)

REQUIRED_FIGURES = {
    'Fig01_architecture': 'Manual: system architecture diagram',
    'Fig02_E1_label_flip': 'E1 accuracy vs round, all methods, 20% data poisoning, PneumoniaMNIST',
    'Fig03_E1_LIE': 'E1 accuracy vs round, all methods, 20% LIE, PneumoniaMNIST',
    'Fig04_trust': 'E1 trust evolution, 3 benign + 3 malicious clients',
    'Fig05_QAOA_quality': 'E2 QAOA approximation ratio vs depth p',
    'Fig06_detection_ROC': 'E2 detection ROC: QUBO vs Krum vs FLTrust vs L2-norm',
    'Fig07_convergence': 'E3 gradient-norm proxy vs round for f in {0,0.1,0.2,0.3}',
    'Fig08_modality_gain': 'E4 quantum gain per modality with dataset size',
    'Fig09_ablation': 'E5 ablation accuracy bars: config x {data poison, LIE}',
    'Fig10_sensitivity': 'E6 heatmaps alpha-theta and lambda-mu',
    'Fig11_overhead': 'E7 stacked per-component timing',
    'Fig12_scalability': 'E8 accuracy and round time vs K',
    'Fig13_participation': 'E9 accuracy and TPR/FPR vs k',
}


def _plots(figures, f, r, c, t, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    root = Path(root)

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(figures / (name + '.pdf'), dpi=300)
        fig.savefig(figures / (name + '.png'), dpi=300)
        plt.close(fig)

    def curves(df, y, name, group='method', ylabel=None, logy=False):
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        for key, g in df.groupby(group):
            trace = g.groupby('round')[y].agg(['mean', 'std'])
            ax.plot(trace.index, trace['mean'], label=str(key))
            if trace['std'].notna().any():
                ax.fill_between(trace.index, trace['mean'] - trace['std'].fillna(0),
                                trace['mean'] + trace['std'].fillna(0), alpha=0.15)
        ax.set(xlabel='Federated round', ylabel=ylabel or y.replace('_', ' '))
        if logy:
            ax.set_yscale('log')
        ax.legend(fontsize=7)
        save(fig, name)

    # Fig 2/3: E1 primary conditions (20%; LIE z_max=1.0), PneumoniaMNIST, mean +/- std over seeds.
    e1 = r[(r.experiment == 'E1') & (r.fraction == 0.2) & (r.dataset == 'pneumoniamnist')]
    curves(e1[e1.attack == 'label_flip'], 'accuracy', 'Fig02_E1_label_flip', ylabel='Test accuracy')
    curves(e1[(e1.attack == 'lie') & (e1.lie_z == 1)], 'accuracy', 'Fig03_E1_LIE', ylabel='Test accuracy')

    # Fig 4: full trust traces of 3 malicious + 3 benign clients (FedQTrust, E1, 20% LIE, seed 42).
    trace_path = root / 'tables/trust_trace.csv'
    if trace_path.exists():
        tr = pd.read_csv(trace_path)
        tr = tr[(tr.experiment == 'E1') & (tr.method == 'FedQTrust') & (tr.attack == 'lie')
                & (tr.fraction == 0.2) & (tr.lie_z == 1) & (tr.seed == 42)]
        if not tr.empty:
            rank = tr.groupby('client').selected.sum().sort_values(ascending=False)
            mal = [cid for cid in rank.index if tr[tr.client == cid].is_malicious.iloc[0]][:3]
            ben = [cid for cid in rank.index if not tr[tr.client == cid].is_malicious.iloc[0]][:3]
            fig, ax = plt.subplots(figsize=(8, 5))
            for cid in mal + ben:
                g = tr[tr.client == cid].sort_values('round')
                kind = 'malicious' if cid in mal else 'benign'
                ax.plot(g['round'], g.trust, '--' if cid in mal else '-', label=f'client {cid} ({kind})')
            theta = tr.theta.iloc[0]
            ax.axhline(theta, color='k', lw=0.8, ls=':', label=f'theta={theta}')
            ax.set(xlabel='Federated round', ylabel='Trust score tau')
            ax.legend(fontsize=7)
            save(fig, 'Fig04_trust')

    # Fig 5: approximation ratio vs p.
    bench = root / 'benchmarks/E2/optimizer_trials.csv'
    if bench.exists():
        b = pd.read_csv(bench)
        if 'item' in b:
            b = b[b.item == 'approximation_ratio']
        b = b[(b.solver == 'QAOA') & ~b.noise.astype(bool)]
        if not b.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            for n, g in b.groupby('n'):
                tr2 = g.groupby('p').objective_ratio.agg(['mean', 'std'])
                ax.errorbar(tr2.index, tr2['mean'], yerr=tr2['std'].fillna(0), fmt='o-', capsize=3, label=f'|E|={n}')
            ax.set(xlabel='QAOA depth p', ylabel='Approximation ratio (QAOA / exact optimum)')
            ax.set_xticks(sorted(b.p.unique()))
            ax.legend()
            save(fig, 'Fig05_QAOA_quality')

    # Fig 6: detection ROC for every E2 selection method.
    roc = c[(c.experiment == 'E2') & c.attack_active & ~c.noise.astype(bool)]
    if not roc.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        names = {'No-QUBO': 'L2-norm', 'FedQTrust': 'QUBO'}
        for (method, solver), g in roc.groupby(['method', 'solver']):
            if g.is_malicious.nunique() < 2:
                continue
            x, y, _ = roc_curve(g.is_malicious, g.detection_score)
            label = names.get(method, method) + (f' ({solver.upper()})' if method == 'FedQTrust' else '')
            ax.plot(x, y, label=label)
            plotted = True
        if plotted:
            ax.plot([0, 1], [0, 1], 'k:', lw=0.8)
            ax.set(xlabel='False-positive rate', ylabel='True-positive rate')
            ax.legend()
            save(fig, 'Fig06_detection_ROC')
        else:
            plt.close(fig)

    # Fig 7: gradient-norm proxy for Byzantine fractions (lambda=1); 7b: lambda sweep under 20% LIE.
    e3 = r[(r.experiment == 'E3') & (r.method == 'FedQTrust')]
    curves(e3[e3.lambda_ == 1].assign(f=lambda d: 'f=' + d.fraction.astype(str)), 'gradient_proxy',
           'Fig07_convergence', group='f', ylabel='Gradient-norm proxy Delta(t)', logy=True)
    curves(e3[(e3.attack == 'lie') & (e3.fraction == 0.2)].assign(lam=lambda d: 'lambda=' + d.lambda_.astype(str)),
           'gradient_proxy', 'Fig07b_convergence_lambda', group='lam', ylabel='Gradient-norm proxy Delta(t)', logy=True)

    # Fig 8: quantum gain per dataset; dataset size on secondary axis.
    gain = root / 'tables/modality_gain.csv'
    if gain.exists():
        g = pd.read_csv(gain).query("attack=='benign'").groupby('dataset').difference.agg(['mean', 'std'])
        order = [d for d in ('pathmnist', 'octmnist', 'pneumoniamnist', 'retinamnist', 'breastmnist') if d in g.index]
        g = g.loc[order]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(g)), g['mean'] * 100, yerr=g['std'].fillna(0) * 100, capsize=3)
        ax.set_xticks(range(len(g)), g.index, rotation=25)
        ax.set(ylabel='Quantum gain: FedQTrust - PQS-BFL (pp)')
        sizes = {}
        for path in (root / 'partitions').glob('*.json'):
            partition = json.loads(path.read_text())
            sizes[partition['dataset']] = len(partition['root_indices']) + sum(
                (len(x['train']) + len(x['validation']) for x in partition['clients']))
        if all((d in sizes for d in g.index)):
            right = ax.twinx()
            right.plot(range(len(g)), [sizes[d] for d in g.index], 'ko--', label='Training samples')
            right.set_ylabel('Training samples (log)')
            right.set_yscale('log')
            right.legend(loc='upper right')
        save(fig, 'Fig08_modality_gain')

    # Fig 9: grouped bars, each ablation config x {data poisoning, LIE}, PneumoniaMNIST.
    e5 = f[(f.experiment == 'E5') & (f.dataset == 'pneumoniamnist')]
    if not e5.empty:
        configs = [m for m in ('No-QUBO', 'No-Trust', 'No-Blockchain', 'No-PQC', 'No-Quantum', 'FedQTrust') if m in set(e5.method)]
        fig, ax = plt.subplots(figsize=(9, 5))
        for i, (attack, label) in enumerate((('label_flip', '20% data poisoning'), ('lie', '20% LIE'))):
            stats = e5[e5.attack == attack].groupby('method').accuracy.agg(['mean', 'std']).reindex(configs)
            ax.bar(np.arange(len(configs)) + (i - 0.5) * 0.38, stats['mean'] * 100, 0.38,
                   yerr=stats['std'].fillna(0) * 100, capsize=3, label=label)
        ax.set_xticks(range(len(configs)), configs, rotation=20)
        ax.set(ylabel='Test accuracy (%)')
        ax.legend()
        save(fig, 'Fig09_ablation')

    # Fig 10: sensitivity heatmaps, other parameters fixed at defaults.
    g = f[f.experiment == 'E6']
    if not g.empty:
        from .plan import Job
        defaults = asdict(Job())
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, (x, y) in zip(axes, [('alpha', 'theta'), ('lambda_', 'mu')]):
            part = g
            for field in ('alpha', 'theta', 'lambda_', 'mu', 'p', 'reputation_decay', 'window'):
                if field not in (x, y):
                    part = part[np.isclose(part[field], defaults[field])]
            pivot = part.pivot_table(index=y, columns=x, values='accuracy') * 100
            im = ax.imshow(pivot.values, aspect='auto', origin='lower')
            ax.set_xticks(range(len(pivot.columns)), [f'{v:g}' for v in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [f'{v:g}' for v in pivot.index])
            for (i, j), v in np.ndenumerate(pivot.values):
                if np.isfinite(v):
                    ax.text(j, i, f'{v:.1f}', ha='center', va='center', fontsize=7, color='w')
            ax.set(xlabel=x.rstrip('_'), ylabel=y.rstrip('_'))
            fig.colorbar(im, ax=ax, label='Test accuracy (%)')
        save(fig, 'Fig10_sensitivity')

    # Fig 11: stacked per-component timing per round (ms).
    g = t[t.experiment == 'E7']
    if not g.empty:
        cols = {'training_s': 'Local training', 'selection_s': 'Client selection (QAOA)', 'blockchain_s': 'Ledger',
                'kem_aes_s': 'ML-KEM-768 + AES-GCM', 'signature_s': 'ML-DSA-65 sign/verify', 'aggregation_s': 'Aggregation'}
        cols = {k: v for k, v in cols.items() if k in g}
        data = (g.groupby('method')[list(cols)].mean() * 1000).rename(columns=cols)
        ax = data.plot.bar(stacked=True, figsize=(9, 5))
        ax.set(ylabel='Milliseconds per round', xlabel='')
        ax.tick_params(axis='x', rotation=0)
        save(ax.figure, 'Fig11_overhead')

    # Fig 12/13: E8 (vs K) and E9 (vs k).
    detection_path = root / 'tables/detection.csv'
    detection = pd.read_csv(detection_path) if detection_path.exists() and detection_path.stat().st_size > 1 else pd.DataFrame()
    for exp, name, x in [('E8', 'Fig12_scalability', 'clients'), ('E9', 'Fig13_participation', 'k')]:
        data = f[f.experiment == exp]
        if data.empty:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for (method, attack), part in data.groupby(['method', 'attack']):
            label = f'{method}/{attack}'
            accuracy = part.groupby(x).accuracy.mean() * 100
            axes[0].plot(accuracy.index, accuracy.values, 'o-', label=label)
            times = t[t.job_id.isin(part.job_id)].groupby(x)
            selection = times.selection_s.mean()
            axes[2].plot(selection.index, selection.values, 'o-', label=label)
            if exp == 'E8':
                total = times.total_s.mean()
                axes[1].plot(total.index, total.values, 'o-', label=label)
            elif not detection.empty:
                joined = detection.merge(part[['job_id', x]].drop_duplicates(), on='job_id')
                for rate in ['tpr', 'fpr']:
                    values = joined.groupby(x)[rate].mean()
                    axes[1].plot(values.index, values.values, 'o-', label=label + '/' + rate.upper())
        for ax, y in zip(axes, ['Test accuracy (%)', 'Round time (s)' if exp == 'E8' else 'TPR / FPR',
                                'QAOA selection time (s)']):
            ax.set(xlabel='K (total clients)' if x == 'clients' else 'k (clients per round)', ylabel=y)
            ax.set_xticks(sorted(data[x].unique()))
            ax.legend(fontsize=6)
        save(fig, name)

    # Fig 1: architecture (the manuscript marks it 'Manual'; this is an auto-drawn placeholder).
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis('off')
    labels = [
        '1. Server reads trust scores from the ledger',
        '2. Trust pre-filter + QUBO client selection (QAOA; SA fallback)',
        '3. Selected clients: E local epochs of FedQCNN',
        '4. ML-KEM-768 + AES-GCM update transport; ML-DSA-65 signatures',
        '5. Trust-weighted aggregation (shared backbone, per-modality heads)',
        '6. Server signs global model; trust scores written to the ledger'
    ]
    for i, label in enumerate(labels):
        y = 0.93 - i * 0.16
        ax.text(0.5, y, label, ha='center', va='center', bbox=dict(boxstyle='round', fc='#e7effa'))
        if i < len(labels) - 1:
            ax.annotate('', xy=(0.5, y - 0.12), xytext=(0.5, y - 0.035), arrowprops=dict(arrowstyle='->'))
    save(fig, 'Fig01_architecture')

def _latex(root, summary, limitations):
    def escape(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')

    audit = json.loads((root / 'audit.json').read_text())
    lines = [
        r'\documentclass{article}',
        r'\usepackage[margin=20mm]{geometry}',
        r'\usepackage{graphicx,longtable,booktabs,xcolor,pdflscape,amsmath}',
        r'\begin{document}',
        r'\title{FedQTrust: Measured Experimental Results}\maketitle',
        r'\section{Scope and completion}',
        f"Completed training conditions: {audit['completed']} of {audit['expected']}.",
        'All numerical results below are measured. Standard deviations describe independent seeds within the same condition. Accuracy and F1 are proportions between zero and one.',
        r'\textcolor{red}{Incomplete coverage and unresolved protocol qualifications must be reviewed before making publication claims.}',
        r'\section{Proposed architecture and implemented method}',
        r'Each grayscale $28\times28$ image is divided into non-overlapping $2\times2$ patches. A four-qubit ZZFeatureMap with one repetition encodes each patch using feature angles. A RealAmplitudes circuit with two repetitions and linear entanglement has twelve trainable circuit parameters. Four Pauli-Z expectations per patch produce 784 features. The circuit is simulated exactly as a 16-amplitude statevector in PyTorch with analytic gradients; a unit test verifies outputs and gradients agree with Qiskit EstimatorQNN/TorchConnector (parameter shift) to numerical precision.',
        r'The shared classifier is $784\rightarrow512\rightarrow256$ with ReLU activations and separate output heads for the five datasets. Classical controls replace the quantum patch operation with a stride-two convolution. The specified dense layers give approximately half a million parameters; this does not reproduce the manuscript claim of 8K parameters.',
        r'Client trust combines held-out accuracy, the sliding cosine consistency of observed updates, and decayed reputation: $\tau_i=\max(0.1,0.4a_i+0.3c_i+0.3r_i)$. The default history window is five and reputation decay is 0.9. Trust weights, thresholds, penalties and circuit depth vary only in their explicit sensitivity conditions.',
        r'Selection minimizes $-\sum_i\tau_i x_i+\lambda\sum_{i<j}D_{ij}x_ix_j+\mu(\sum_i x_i-k)^2$. The defaults are $\lambda=1$, $\mu=2$, QAOA depth three, 1,024 shots and 100 optimizer iterations. Selection uses cached prior updates. Cardinality projection and any SA fallback are recorded. The K=80 fallback and solver overrides must be distinguished from QAOA evidence.',
        r'ML-KEM-768 establishes transport keys for HKDF and AES-GCM. ML-DSA-65 authenticates updates and the global model. Paper infrastructure uses Fabric to verify registered signer proofs before accepting updated trust. Local infrastructure keeps real transport cryptography and application signature verification but uses an in-process ledger; it provides no Fabric or consensus measurements. Development mode omits the cryptographic infrastructure. Transport confidentiality does not hide individual updates from the aggregating server.',
        r'\section{Experimental procedure}',
        r'\begin{enumerate}',
        r'\item Use the official MedMNIST training and test splits. Convert images to grayscale tensors. Hold out one percent of training data for FLTrust root updates for every method, ensuring matched data access.',
        r'\item Partition the remaining data using Dirichlet concentration 0.5 and partition seed 42. Reserve twenty percent within each client for validation. Deterministic minimum-size repair preserves every original example and its global label.',
        r'\item Initialize each method with the assigned model seed. Main comparisons use seeds 42, 123, 456, 789 and 999. E1 distributes forty clients across five modalities; single-modality conditions distribute all clients over that dataset.',
        r'\item In each round, select five clients using the assigned method, then train each for three local Adam epochs with batch size 32, weight decay $10^{-4}$ and learning rate $0.01/\sqrt{t}$. FedProx selects its proximal coefficient using a held-out validation warm-up.',
        r'\item Activate the configured attack from round ten. Apply label corruption, sign reversal, Gaussian perturbation, adaptive LIE or free riding as specified by the condition. Combined attacks use disjoint client groups. Defenses do not receive the evaluator\textquotesingle s malicious-client identities.',
        r'\item Authenticate and transport selected updates, evaluate local validation accuracy, update trust and aggregate. Shared parameters combine contributions across modalities; each classification head combines only matching clients. Save the model, trust history, update cache and random states after every completed round.',
        r'\item Evaluate the global model on official test data each round. Report final-round results, without choosing a checkpoint using test accuracy. Default runs use one hundred rounds; E3 uses two hundred and the dedicated E7 timing conditions use twenty.',
        r'\item Compare matched seeds, retain raw observations and report means, sample standard deviations, paired Wilcoxon statistics and the declared multiplicity correction. Missing experiments remain missing; no numerical target is substituted for a measurement.',
        r'\end{enumerate}',
        r'\section{Experiment coverage}',
        'E1 evaluates baselines and attacks; E2 evaluates selection solvers and noisy circuits; E3 measures convergence; E4 compares modalities; E5 removes components; E6 varies hyperparameters; E7 measures component overhead; E8 varies client count; E9 varies selected-client count. Dedicated E2 and E7 benchmarks are additional to training jobs.',
        r'\section{Measured results}',
    ]
    parameters = ['fraction', 'clients', 'k', 'rounds', 'label_fraction', 'sigma', 'lie_z',
                  'alpha', 'beta', 'gamma', 'theta', 'lambda_', 'mu', 'p',
                  'reputation_decay', 'window', 'solver', 'noise', 'infrastructure']
    for experiment, group in summary.groupby('experiment'):
        lines.extend([r'\subsection{' + experiment + '}',
                      'The condition column identifies settings that vary within this experiment; other settings follow the procedure above.',
                      r'\begin{landscape}\small'])
        varied = [key for key in parameters if key in (
            'fraction',
            'solver',
            'infrastructure'
        ) or group[key].nunique(dropna=False) > 1]
        table = group.copy()
        table['condition'] = table.apply(lambda row: str(row['attack']) + '; ' + '; '.join(
            f'{key}={row[key]}' for key in varied), axis=1)
        columns = ['method', 'dataset', 'condition', 'accuracy_mean', 'accuracy_std',
                   'weighted_f1_mean', 'accuracy_count']
        table = table[columns].rename(columns={'accuracy_mean': 'Acc. mean', 'accuracy_std': 'Acc. SD',
                                               'weighted_f1_mean': 'F1 mean', 'accuracy_count': 'Seeds'})
        lines.append(table.to_latex(index=False, longtable=True, escape=True,
                                   column_format='p{2.2cm}p{2.5cm}p{9cm}rrrr',
                                   float_format=lambda value: f'{value:.4f}'))
        lines.append(r'\end{landscape}\normalsize')
    lines.extend([r'\section{Interpretation and limitations}', r'\begin{itemize}'])
    lines.extend(r'\item \textcolor{red}{' + escape(value) + '}' for value in limitations)
    lines.extend([r'\end{itemize}', r'\section{Figures}'])
    for path in sorted((root / 'figures').glob('*.pdf')):
        lines.extend([r'\begin{figure}[p]\centering',
                      r'\includegraphics[width=\linewidth,height=.8\textheight,keepaspectratio]{figures/' + path.name + '}',
                      r'\caption{' + escape(path.stem.replace('_', ' ')) + r'.}\end{figure}'])
    lines.append(r'\end{document}')
    (root / 'main.tex').write_text('\n'.join(lines))


# ---------------------------------------------------------------------------
# Manuscript-format tables (Tables "main_results", "ablation_results", "overhead"
# of main-v2.tex). Cells are filled ONLY from measured job outputs; a condition
# that has not completed stays "--", exactly as in the manuscript placeholders.
# ---------------------------------------------------------------------------
PAPER_METHODS = ['FedAvg', 'FedProx', 'FedQCNN', 'Krum', 'FLTrust', 'PQS-BFL', 'SecEdge-MC', 'FedQTrust']
PAPER_ATTACK_ROWS = [
    ('Benign', 'benign', 0.0),
    ('Label Flip 20\\%', 'label_flip', 0.2),
    ('Sign-Flip 20\\%', 'sign_flip', 0.2),
    ('LIE 20\\%', 'lie', 0.2),
    ('Free-Ride 40\\%', 'free_riding', 0.4),
    ('Combined 30\\%', 'combined', 0.3),
]
ABLATION_ROWS = [('No-QUBO (L2-norm)', 'No-QUBO'), ('No-Trust', 'No-Trust'), ('No-Blockchain', 'No-Blockchain'),
                 ('No-PQC', 'No-PQC'), ('No-Quantum', 'No-Quantum'), ('\\textbf{FedQTrust}', 'FedQTrust')]


def _stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '$^{**}$' if p < 0.01 else ('$^{*}$' if p < 0.05 else '')


def _paired_p(per_seed, method):
    if method == 'FedQTrust' or 'FedQTrust' not in per_seed or method not in per_seed:
        return None
    paired = per_seed[['FedQTrust', method]].dropna()
    if len(paired) < 2:
        return None
    diff = paired['FedQTrust'] - paired[method]
    return 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, method='exact').pvalue)


def paper_tables(root, macro, f, t, tables):
    out = Path(root) / 'paper_tables'
    out.mkdir(exist_ok=True)
    cells = []
    # --- Table: main_results (E1, macro-average over 5 datasets, 5 seeds) -------------
    e1 = macro[(macro.experiment == 'E1') & ((macro.attack != 'lie') | (macro.lie_z == 1.0))
               & ((macro.attack != 'gaussian'))]
    lines = [r'\begin{tabular}{lcccccccc}', r'\toprule',
             r'\textbf{Attack} & \textbf{FedAvg} & \textbf{FedProx} & \textbf{FedQCNN} & \textbf{Krum} & '
             r'\textbf{FLTrust} & \textbf{PQS-BFL} & \textbf{SecEdge} & \textbf{FedQTrust} \\', r'\midrule']
    for label, attack, fraction in PAPER_ATTACK_ROWS:
        part = e1[(e1.attack == attack) & np.isclose(e1.fraction, fraction)]
        per_seed = part.pivot_table(index='seed', columns='method', values='accuracy')
        row = [label]
        for method in PAPER_METHODS:
            vals = per_seed[method].dropna() if method in per_seed else pd.Series(dtype=float)
            if vals.empty:
                cell = '--'
            else:
                p = _paired_p(per_seed, method)
                cell = f'{100 * vals.mean():.2f} $\\pm$ {100 * vals.std(ddof=1) if len(vals) > 1 else 0:.2f}{_stars(p)}'
                cells.append(dict(table='main_results', row=label, column=method, mean=vals.mean(),
                                  std=vals.std(ddof=1) if len(vals) > 1 else np.nan, n_seeds=len(vals), p_vs_fedqtrust=p))
            row.append(r'\textbf{' + cell + '}' if method == 'FedQTrust' and cell != '--' else cell)
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_main_results.tex').write_text('\n'.join(lines) + '\n')

    # --- Table: ablation_results (E5, PneumoniaMNIST, 5 seeds) -------------------------
    e5 = f[(f.experiment == 'E5') & (f.dataset == 'pneumoniamnist')]
    e5t = t[(t.experiment == 'E5') & t.job_id.isin(e5.job_id)]
    lines = [r'\begin{tabular}{lccccc}', r'\toprule',
             r'\textbf{Config} & \multicolumn{2}{c}{\textbf{Data Poisoning}} & \multicolumn{2}{c}{\textbf{LIE Attack}} & \textbf{Round Time} \\',
             r' & Acc (\%) & F1 & Acc (\%) & F1 & (ms) \\', r'\midrule']
    for label, method in ABLATION_ROWS:
        row = [label]
        for attack in ('label_flip', 'lie'):
            part = e5[(e5.method == method) & (e5.attack == attack)]
            if part.empty:
                row += ['--', '--']
                continue
            row += [f'{100 * part.accuracy.mean():.2f} $\\pm$ {100 * part.accuracy.std(ddof=1) if len(part) > 1 else 0:.2f}',
                    f'{part.weighted_f1.mean():.3f}']
            cells.append(dict(table='ablation_results', row=method, column=attack, mean=part.accuracy.mean(),
                              std=part.accuracy.std(ddof=1), n_seeds=len(part), f1=part.weighted_f1.mean()))
        times = e5t[e5t.method == method].total_s
        row.append(f'{1000 * times.mean():.0f}' if len(times) else '--')
        lines.append(' & '.join(row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_ablation_results.tex').write_text('\n'.join(lines) + '\n')

    # --- Table: overhead (E7 timing jobs, per round) -----------------------------------
    e7 = t[t.experiment == 'E7']
    q = e7[e7.method == 'FedQTrust']
    base = e7[e7.method == 'FedAvg']
    lines = [r'\begin{tabular}{lcc}', r'\toprule', r'\textbf{Component} & \textbf{Time (ms)} & \textbf{\% of Round} \\', r'\midrule']
    components = [('FedQCNN Training ($k=5$ clients)', 'training_s'), ('QAOA Client Selection', 'selection_s'),
                  ('Blockchain Commit', 'blockchain_s'), ('ML-KEM-768 Enc/Dec', 'kem_aes_s'),
                  ('ML-DSA-65 Sign/Verify', 'signature_s')]
    for label, col in components:
        if q.empty or col not in q:
            lines.append(f'{label} & -- & -- \\\\')
            continue
        ms = 1000 * q[col].mean()
        share = 100 * q[col].mean() / q.total_s.mean()
        lines.append(f'{label} & {ms:.1f} & {share:.1f}\\% \\\\')
        cells.append(dict(table='overhead', row=label, column='ms', mean=ms, share_percent=share))
    lines.append(r'\midrule')
    if not q.empty and not base.empty:
        extra = 1000 * (q.total_s.mean() - base.total_s.mean())
        pct = 100 * (q.total_s.mean() / base.total_s.mean() - 1)
        lines.append(f'\\textbf{{Total Overhead vs.\\ FedAvg}} & {extra:.1f} & \\textbf{{{pct:.1f}\\%}} \\\\')
        cells.append(dict(table='overhead', row='total_vs_fedavg', column='percent', mean=pct))
    else:
        lines.append(r'\textbf{Total Overhead vs.\ FedAvg} & -- & \textbf{--\%} \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (out / 'tab_overhead.tex').write_text('\n'.join(lines) + '\n')
    pd.DataFrame(cells).to_csv(out / 'paper_table_cells.csv', index=False)


def _pm(values, scale=100.0, digits=2):
    values = pd.Series(values).dropna()
    if values.empty:
        return '--'
    sd = values.std(ddof=1) * scale if len(values) > 1 else 0.0
    return f'{values.mean() * scale:.{digits}f} $\\pm$ {sd:.{digits}f}'


def _write_tex(df, path, caption, colspec=None):
    """booktabs table; '--' marks conditions with no completed measurement."""
    df = df.fillna('--')
    colspec = colspec or 'l' * len(df.columns)
    lines = [f'% {caption}', r'\begin{tabular}{' + colspec + '}', r'\toprule',
             ' & '.join(r'\textbf{' + str(col).replace('_', r'\_').replace('%', r'\%').replace('|', r'$|$') + '}' for col in df.columns) + r' \\',
             r'\midrule']
    def cell(v):
        v = str(v)
        return v if '$' in v else v.replace('_', r'\_').replace('%', r'\%').replace('|', r'$|$')
    for row in df.itertuples(index=False):
        lines.append(' & '.join(cell(v) for v in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    path.write_text('\n'.join(lines) + '\n')
    df.to_csv(path.with_suffix('.csv'), index=False)


def experiment_tables(root, f, r, t, macro):
    """Every 'WHAT TO REPORT' item of E1-E9 in main-v2.tex as .tex + .csv."""
    root = Path(root)
    out = root / 'paper_tables'
    out.mkdir(exist_ok=True)
    tables = root / 'tables'

    # E1 per primary dataset (report individually), same layout as Table main_results.
    for dataset in ('pneumoniamnist', 'pathmnist'):
        part = f[(f.experiment == 'E1') & (f.dataset == dataset)]
        rows = []
        for label, attack, fraction in PAPER_ATTACK_ROWS:
            cond = part[(part.attack == attack) & np.isclose(part.fraction, fraction)
                        & ((part.attack != 'lie') | (part.lie_z == 1.0))]
            per_seed = cond.pivot_table(index='seed', columns='method', values='accuracy')
            row = {'Attack': label}
            for method in PAPER_METHODS:
                row[method] = (_pm(per_seed[method]) + _stars(_paired_p(per_seed, method))) if method in per_seed else '--'
            rows.append(row)
        _write_tex(pd.DataFrame(rows), out / f'tab_main_results_{dataset}.tex',
                   f'E1 test accuracy (%) mean +/- std over seeds, {dataset}', 'l' + 'c' * len(PAPER_METHODS))

    # E1 supplementary: 10%/30% ratios, Gaussian sigma sweep, LIE z_max sweep (macro-average).
    e1 = macro[macro.experiment == 'E1']
    rows = []
    for (attack, fraction, sigma, z), g in e1.groupby(['attack', 'fraction', 'sigma', 'lie_z'], dropna=False):
        if attack != 'gaussian' and sigma != 0.5:
            continue
        if attack != 'lie' and z != 1.0:
            continue
        row = {'Attack': attack.replace('_', ' '), 'r': f'{fraction:g}',
               'Param': f'sigma={sigma:g}' if attack == 'gaussian' else (f'z={z:g}' if attack == 'lie' else '')}
        per_seed = g.pivot_table(index='seed', columns='method', values='accuracy')
        for method in PAPER_METHODS:
            row[method] = _pm(per_seed[method]) if method in per_seed else '--'
        rows.append(row)
    if rows:
        _write_tex(pd.DataFrame(rows), out / 'tab_e1_supplementary.tex',
                   'E1 all attack strengths, macro-average accuracy (%)', 'lll' + 'c' * len(PAPER_METHODS))

    # Accuracy at rounds 10, 20, 30, 50, 75, 100 (standard reporting requirement).
    checkpoints = [10, 20, 30, 50, 75, 100]
    sub = r[r['round'].isin(checkpoints)]
    if not sub.empty:
        acc = (sub.groupby(['experiment', 'method', 'attack', 'fraction', 'dataset', 'seed', 'round'])
               .accuracy.mean().reset_index())
        wide = acc.groupby(['experiment', 'method', 'attack', 'fraction', 'dataset', 'round']).accuracy.mean().unstack('round') * 100
        wide.reset_index().to_csv(out / 'accuracy_at_rounds.csv', index=False)
        prim = wide.reset_index()
        prim = prim[(prim.experiment == 'E1') & (prim.dataset == 'pneumoniamnist') & np.isclose(prim.fraction, 0.2)]
        if not prim.empty:
            prim = prim.drop(columns=['experiment', 'fraction', 'dataset'])
            prim.columns = [f'R{c}' if isinstance(c, (int, np.integer)) else c for c in prim.columns]
            _write_tex(prim.round(2), out / 'tab_accuracy_at_rounds.tex',
                       'E1 PneumoniaMNIST, 20% attacks: test accuracy (%) at rounds 10-100')

    # E2: selection methods under 20% LIE (PathMNIST) + QAOA quality / time / noise.
    detection = pd.read_csv(tables / 'detection.csv') if (tables / 'detection.csv').exists() and (tables / 'detection.csv').stat().st_size > 1 else pd.DataFrame()
    e2 = f[f.experiment == 'E2']
    if not e2.empty:
        rows = []
        for (method, solver, noise), g in e2.groupby(['method', 'solver', 'noise']):
            d = detection[detection.job_id.isin(g.job_id)] if not detection.empty else pd.DataFrame()
            rows.append({'Selection': {'No-QUBO': 'L2-norm', 'FedQTrust': 'QUBO'}.get(method, method)
                         + (f' ({solver.upper()})' if method == 'FedQTrust' else '') + (' + noise' if noise else ''),
                         'Acc (%)': _pm(g.accuracy), 'F1': f'{g.weighted_f1.mean():.3f}',
                         'TPR': f'{d.tpr.mean():.3f}' if not d.empty else '--',
                         'FPR': f'{d.fpr.mean():.3f}' if not d.empty else '--'})
        _write_tex(pd.DataFrame(rows), out / 'tab_e2_selection.tex', 'E2 selection methods, 20% LIE, PathMNIST', 'lcccc')
    bench = root / 'benchmarks/E2/optimizer_trials.csv'
    if bench.exists():
        b = pd.read_csv(bench)
        if 'item' in b:
            ratio = b[(b.item == 'approximation_ratio') & (b.solver == 'QAOA')]
            if not ratio.empty:
                tab = ratio.groupby(['n', 'p']).objective_ratio.agg(['mean', 'std']).reset_index()
                tab = tab.assign(ratio=lambda d: d['mean'].map('{:.3f}'.format) + ' $\\pm$ ' + d['std'].fillna(0).map('{:.3f}'.format))
                _write_tex(tab.pivot(index='n', columns='p', values='ratio').add_prefix('p=').reset_index().rename(columns={'n': '|E|'}),
                           out / 'tab_e2_approximation_ratio.tex', 'E2 QAOA approximation ratio (5 instances)')
            timing = b[b.item == 'timing']
            if not timing.empty:
                tab = timing.groupby(['n', 'solver']).runtime_s.agg(['mean', 'std']).reset_index()
                tab['ms'] = (tab['mean'] * 1000).map('{:.1f}'.format) + ' $\\pm$ ' + (tab['std'].fillna(0) * 1000).map('{:.1f}'.format)
                _write_tex(tab.pivot(index='n', columns='solver', values='ms').reset_index().rename(columns={'n': '|E|'}),
                           out / 'tab_e2_solver_time.tex', 'E2 solve time (ms), 10 trials')
            noise = b[b.item.isin(['approximation_ratio', 'noise']) & (b.p == 3) & (b.solver == 'QAOA')]
            if noise.noise.astype(bool).any():
                tab = noise.groupby(['n', 'noise']).objective_ratio.mean().unstack('noise')
                tab.columns = ['noiseless' if not c else 'fake_nairobi' for c in tab.columns]
                _write_tex(tab.round(3).reset_index().rename(columns={'n': '|E|'}), out / 'tab_e2_noise.tex',
                           'E2 approximation ratio, noiseless vs Nairobi noise (p=3)')

    # E3: fit a/sqrt(t)+b, R^2 and rounds-to-80%.
    fits = tables / 'convergence_fits.csv'
    if fits.exists() and fits.stat().st_size > 1:
        fit = pd.read_csv(fits)
        meta = f[f.experiment == 'E3'][['job_id', 'method', 'attack', 'fraction', 'lambda_']].drop_duplicates()
        fit = meta.merge(fit, on='job_id')
        fit = fit[['method', 'attack', 'fraction', 'lambda_', 'a', 'b', 'r2', 'rounds_to_80']]
        fit = fit.rename(columns={'fraction': 'f', 'lambda_': 'lambda', 'r2': 'R2', 'rounds_to_80': 'rounds to 80%'})
        _write_tex(fit.round(4), out / 'tab_e3_convergence.tex', 'E3 convergence fit and rounds to 80% accuracy')

    # E4: quantum gain per dataset and condition + Pearson r.
    gain = tables / 'modality_gain.csv'
    if gain.exists():
        g = pd.read_csv(gain)
        tab = g.groupby(['dataset', 'attack']).difference.agg(['mean', 'std']).reset_index()
        tab['gain (pp)'] = (tab['mean'] * 100).map('{:.2f}'.format) + ' $\\pm$ ' + (tab['std'].fillna(0) * 100).map('{:.2f}'.format)
        tab = tab.pivot(index='dataset', columns='attack', values='gain (pp)').reset_index()
        corr = tables / 'modality_size_correlation.json'
        if corr.exists():
            c = json.loads(corr.read_text())
            tab = pd.concat([tab, pd.DataFrame([{'dataset': f"Pearson r={c['r']:.3f}, p={c['p']:.3g}"}])])
        _write_tex(tab, out / 'tab_e4_quantum_gain.tex', 'E4 quantum gain FedQTrust - PQS-BFL (pp)')

    # E5: TPR/FPR per configuration (final trust tau < theta).
    spec = tables / 'detection_final_trust.csv'
    if spec.exists() and spec.stat().st_size > 1:
        d = pd.read_csv(spec)
        d5 = d[d.experiment == 'E5']
        if not d5.empty:
            tab = d5.groupby(['method', 'attack'])[['tpr', 'fpr', 'pearson_rho']].mean().round(3).reset_index()
            _write_tex(tab, out / 'tab_e5_detection.tex', 'E5 malicious detection (final trust < theta)')
        d1 = d[d.experiment == 'E1']
        if not d1.empty:
            tab = d1.groupby(['method', 'attack', 'fraction', 'attack_class'])[['tpr', 'fpr', 'pearson_rho']].mean().round(3).reset_index()
            _write_tex(tab, out / 'tab_e1_detection.tex', 'E1 detection TPR/FPR and trust-reliability rho, poisoning vs free-riding')

    # E6: one-at-a-time sensitivity.
    e6 = f[f.experiment == 'E6']
    if not e6.empty:
        from .plan import Job
        defaults = asdict(Job())
        params = ('alpha', 'theta', 'lambda_', 'mu', 'p', 'reputation_decay', 'window')
        rows = []
        for param in params:
            others = [q for q in params if q != param]
            part = e6
            for q in others:
                part = part[np.isclose(part[q], defaults[q])]
            for value, g in part.groupby(param):
                rows.append({'Parameter': param.rstrip('_'), 'Value': f'{value:g}', 'Acc (%)': f'{100 * g.accuracy.mean():.2f}',
                             'Delta vs default (pp)': ''})
        tab = pd.DataFrame(rows)
        if not tab.empty:
            base = e6
            for q in params:
                base = base[np.isclose(base[q], defaults[q])]
            if not base.empty:
                ref = 100 * base.accuracy.mean()
                tab['Delta vs default (pp)'] = (tab['Acc (%)'].astype(float) - ref).map('{:+.2f}'.format)
            _write_tex(tab, out / 'tab_e6_sensitivity.tex', 'E6 single-parameter sensitivity (seed 42)', 'llcc')

    # E7: component micro-benchmarks + communication.
    comp = root / 'benchmarks/E7/component_summary.csv'
    if comp.exists():
        cs = pd.read_csv(comp)
        cs = cs.assign(**{'mean (ms)': (cs.mean_s * 1000).round(3), 'std (ms)': (cs.std_s * 1000).round(3),
                          'p50 (ms)': (cs.p50_s * 1000).round(3), 'p99 (ms)': (cs.p99_s * 1000).round(3)})
        _write_tex(cs[['component', 'n', 'mean (ms)', 'std (ms)', 'p50 (ms)', 'p99 (ms)']].rename(columns={'n': 'trials'}),
                   out / 'tab_e7_components.tex', 'E7 per-component timing')
    local = root / 'benchmarks/E7/local_training_trials.csv'
    if local.exists():
        lt = pd.read_csv(local).groupby('dataset').seconds.agg(['mean', 'std']).reset_index()
        _write_tex(lt.round(3).rename(columns={'mean': 'mean (s)', 'std': 'std (s)'}), out / 'tab_e7_local_training.tex',
                   'E7 FedQCNN local training, 1 round (3 epochs, batch 32)')
    e7 = t[t.experiment == 'E7']
    if not e7.empty:
        comm = e7.groupby('method')[['update_bytes', 'signature_bytes', 'wire_bytes']].mean().reset_index()
        comm = comm.assign(**{'update KB/client': (comm.update_bytes / 5 / 1024).round(1),
                              'signature B/round': comm.signature_bytes.round(0),
                              'wire KB/round': (comm.wire_bytes / 1024).round(1)})
        _write_tex(comm[['method', 'update KB/client', 'signature B/round', 'wire KB/round']], out / 'tab_e7_communication.tex',
                   'E7 communication per round (k=5)')

    # E8 / E9 summary tables.
    for exp, x, name in (('E8', 'clients', 'tab_e8_scalability'), ('E9', 'k', 'tab_e9_clients_per_round')):
        data = f[f.experiment == exp]
        if data.empty:
            continue
        rows = []
        for (method, attack, value), g in data.groupby(['method', 'attack', x]):
            tt = t[t.job_id.isin(g.job_id)]
            d = detection[detection.job_id.isin(g.job_id)] if not detection.empty else pd.DataFrame()
            rows.append({'method': method, 'attack': attack, x: value, 'Acc (%)': _pm(g.accuracy),
                         'selection (s)': f'{tt.selection_s.mean():.3f}', 'round (s)': f'{tt.total_s.mean():.2f}',
                         '|E| mean': f'{tt.eligible.mean():.1f}',
                         'TPR': f'{d.tpr.mean():.3f}' if not d.empty else '--', 'FPR': f'{d.fpr.mean():.3f}' if not d.empty else '--'})
        _write_tex(pd.DataFrame(rows), out / f'{name}.tex', f'{exp} summary')


REQUIRED_TABLES = ['tab_main_results', 'tab_main_results_pneumoniamnist', 'tab_main_results_pathmnist',
                   'tab_e1_supplementary', 'tab_e1_detection', 'tab_accuracy_at_rounds', 'tab_e2_selection',
                   'tab_e2_approximation_ratio', 'tab_e2_solver_time', 'tab_e2_noise', 'tab_e3_convergence',
                   'tab_e4_quantum_gain', 'tab_ablation_results', 'tab_e5_detection', 'tab_e6_sensitivity',
                   'tab_overhead', 'tab_e7_components', 'tab_e7_local_training', 'tab_e7_communication',
                   'tab_e8_scalability', 'tab_e9_clients_per_round']


def output_checklist(root):
    root = Path(root)
    rows = [dict(kind='figure', name=n, description=d, present=(root / 'figures' / (n + '.pdf')).exists())
            for n, d in REQUIRED_FIGURES.items()]
    rows += [dict(kind='table', name=n, description='', present=(root / 'paper_tables' / (n + '.tex')).exists())
             for n in REQUIRED_TABLES]
    pd.DataFrame(rows).to_csv(root / 'tables' / 'required_outputs.csv', index=False)
    return rows
