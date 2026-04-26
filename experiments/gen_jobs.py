#!/usr/bin/env python3
"""
experiments/gen_jobs.py
=======================
Generate sbatch job scripts and phase submit scripts from the YAML configs.

Usage:
    python experiments/gen_jobs.py [--dry-run]

Outputs:
    experiments/jobs/{group_id}/{ticker_set}/{timeframe}.sh   — sbatch scripts
    experiments/submit_phase_{N}.sh                           — submit scripts
"""

import argparse
import os
import stat
import yaml

HERE    = os.path.dirname(os.path.abspath(__file__))
ROOT    = os.path.dirname(HERE)
CONFIGS = os.path.join(HERE, 'configs')
JOBS    = os.path.join(HERE, 'jobs')


def load_configs():
    def _load(name):
        with open(os.path.join(CONFIGS, f'{name}.yaml')) as f:
            return yaml.safe_load(f)
    return _load('common'), _load('ticker_sets'), _load('groups')


SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={log_path}
#SBATCH --error={log_path}

# --- environment ---
cd "{root}"
source .venv/bin/activate

# Prevent OpenBLAS/MKL/OMP from spawning extra threads per worker process.
# Without this, each of the 16 parallel workers tries to create 16 BLAS threads,
# quickly exhausting RLIMIT_NPROC (1000) and causing KeyboardInterrupt in threadpoolctl.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- run ---
python walkforward-compare.py \\
    --strategies {strategies} \\
    --ticker {tickers} \\
    --wf-start {wf_start} \\
    --wf-end {wf_end} \\
    --is-years {is_years} \\
    --oos-years {oos_years} \\
    --step {step} \\
    --n-trials {n_trials} \\
    --objective-metric {objective_metric} \\
    --seed {seed} \\
    --stake {stake} \\
    --cash {cash} \\
    --commission {commission} \\
    --stop-loss {stop_loss} \\
    --take-profit {take_profit} \\
    --wf-max-workers {wf_max_workers} \\
    --out-dir "{out_dir}" \\
{hmm_args_block}
"""


def _hmm_args_block(hmm_args: list) -> str:
    """Format the hmm_args list as continuation lines for the shell command."""
    if not hmm_args:
        return "    --regime-mode strict"   # sensible default if no args provided
    # Pair args and values for cleaner line-wrapping
    parts = []
    i = 0
    while i < len(hmm_args):
        arg = hmm_args[i]
        # Collect all values following this flag (until next flag)
        vals = []
        i += 1
        while i < len(hmm_args) and not hmm_args[i].startswith('--'):
            vals.append(hmm_args[i])
            i += 1
        if vals:
            parts.append(f'    {arg} {" ".join(vals)}')
        else:
            parts.append(f'    {arg}')
    return ' \\\n'.join(parts)


def gen_jobs(dry_run: bool = False):
    common, ticker_sets, groups_cfg = load_configs()

    phases: dict[int, list[str]] = {}   # phase_number -> list of script paths

    for group in groups_cfg['groups']:
        gid         = group['id']
        phase       = group['phase']
        hmm_args    = group.get('hmm_args', [])
        group_ts    = group.get('ticker_sets', ['spy_qqq'])

        for ts_key in group_ts:
            ts_def  = ticker_sets[ts_key]
            tickers = ts_def['tickers']

            for tf in common['timeframes']:
                tf_id    = tf['id']
                is_years = tf['is_years']
                oos_years= tf['oos_years']

                job_name = f'{gid}__{ts_key}__{tf_id}'
                out_dir  = os.path.join(ROOT, 'results', f'phase_{phase}', gid, ts_key, tf_id)
                log_dir  = os.path.join(ROOT, 'logs')
                log_path = os.path.join(log_dir, f'{job_name}_%j.out')

                script = SBATCH_TEMPLATE.format(
                    job_name       = job_name,
                    cpus           = common['cpus_per_task'],
                    log_path       = log_path,
                    root           = ROOT,
                    strategies     = ' '.join(common['strategies']),
                    tickers        = ' '.join(tickers),
                    wf_start       = common['wf_start'],
                    wf_end         = common['wf_end'],
                    is_years       = is_years,
                    oos_years      = oos_years,
                    step           = common['step'],
                    n_trials       = common['n_trials'],
                    objective_metric = common['objective_metric'],
                    seed           = common['seed'],
                    stake          = common['stake'],
                    cash           = common['cash'],
                    commission     = common['commission'],
                    stop_loss      = common['stop_loss'],
                    take_profit    = common['take_profit'],
                    wf_max_workers = common['wf_max_workers'],
                    out_dir        = out_dir,
                    hmm_args_block = _hmm_args_block(hmm_args),
                )

                script_path = os.path.join(JOBS, gid, ts_key, f'{tf_id}.sh')

                if dry_run:
                    print(f'[DRY-RUN] Would write: {script_path}')
                    print(script)
                    print('-' * 60)
                else:
                    os.makedirs(os.path.dirname(script_path), exist_ok=True)
                    os.makedirs(log_dir, exist_ok=True)
                    with open(script_path, 'w') as fh:
                        fh.write(script)
                    os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IEXEC)

                phases.setdefault(phase, []).append(script_path)

    # Write submit_phase_N.sh scripts
    for phase_num, scripts in sorted(phases.items()):
        submit_path = os.path.join(HERE, f'submit_phase_{phase_num}.sh')
        lines = ['#!/bin/bash', f'# Submit all Phase {phase_num} jobs', '']
        for sp in sorted(scripts):
            lines.append(f'sbatch "{sp}"')
        content = '\n'.join(lines) + '\n'

        if dry_run:
            print(f'[DRY-RUN] Would write: {submit_path}')
            print(content)
            print('-' * 60)
        else:
            with open(submit_path, 'w') as fh:
                fh.write(content)
            os.chmod(submit_path, os.stat(submit_path).st_mode | stat.S_IEXEC)
            print(f'Written: {submit_path}  ({len(scripts)} jobs)')

    if not dry_run:
        print(f'\nTotal phases: {len(phases)}')
        total = sum(len(v) for v in phases.values())
        print(f'Total jobs  : {total}')
        print(f'\nJobs written to: {JOBS}')
        print('Submit scripts:')
        for phase_num in sorted(phases):
            print(f'  bash experiments/submit_phase_{phase_num}.sh')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Print generated scripts without writing files')
    args = parser.parse_args()
    gen_jobs(dry_run=args.dry_run)


if __name__ == '__main__':
    main()
