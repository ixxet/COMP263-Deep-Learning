<script lang="ts">
  import { predict } from '$lib/api';
  import type { PredictionResponse, TransactionInput } from '$lib/types';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import ScoreBar from '$lib/components/ScoreBar.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';
  import { browser } from '$app/environment';

  // ── Field definitions ──────────────────────────────────────────────
  const PCA_FIELDS = Array.from({ length: 28 }, (_, i) => `V${i + 1}`);

  const ORDINARY_SAMPLE: TransactionInput = {
    Time: 0, Amount: 43.75,
    ...Object.fromEntries(PCA_FIELDS.map(f => [f, 0]))
  } as TransactionInput;

  const FRAUD_SAMPLE: TransactionInput = {
    Time: 406, Amount: 529,
    V1: -2.31, V2: 1.95,  V3: -1.61, V4: 3.99,  V5: -0.52,
    V6: -1.43, V7: -2.54, V8: 1.39,  V9: -2.77, V10: -2.77,
    V11: 3.2,  V12: -2.9, V13: -0.59,V14: -4.29, V15: 0.39,
    V16: -1.14,V17: -2.83,V18: -0.02,V19: 0.42,  V20: 0.13,
    V21: 0.52, V22: -0.04,V23: -0.47,V24: 0.32,  V25: 0.04,
    V26: 0.18, V27: 0.26, V28: -0.14,
  } as TransactionInput;

  // ── State ──────────────────────────────────────────────────────────
  let transaction: TransactionInput = { ...ORDINARY_SAMPLE };
  let result: PredictionResponse | null = null;
  let loading = false;
  let error = '';
  let scoreMsg = '';
  let showPCA = false;

  // ── Helpers ────────────────────────────────────────────────────────
  function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

  function riskColor(band: string) {
    return band === 'high' ? 'var(--high)' : band === 'uncertain' ? 'var(--uncertain)' : 'var(--accent)';
  }

  function riskBorderColor(band: string) {
    return band === 'high'
      ? 'rgba(239,68,68,.44)'
      : band === 'uncertain'
        ? 'rgba(245,158,11,.44)'
        : 'rgba(20,184,166,.44)';
  }

  function riskExplanation(r: PredictionResponse) {
    if (r.risk_band === 'high')      return 'High-risk case opened for analyst review.';
    if (r.risk_band === 'uncertain') return 'Uncertain — case opened, awaiting agent brief and human review.';
    return 'Low risk — stored for audit only. No case opened.';
  }

  function loadSample(type: 'ordinary' | 'fraud') {
    transaction = type === 'fraud' ? { ...FRAUD_SAMPLE } : { ...ORDINARY_SAMPLE };
    result = null; error = '';
    scoreMsg = type === 'fraud'
      ? 'Loaded known fraud-like sample — significant V14, V4 deviations.'
      : 'Loaded ordinary baseline sample — all PCA features zeroed.';
  }

  // ── Submit ─────────────────────────────────────────────────────────
  async function submit() {
    loading = true; error = ''; result = null; scoreMsg = 'Scoring transaction...';
    try {
      result = await predict(transaction);
      if (browser) {
        localStorage.setItem('fraud-sentinel-model-version', result.model_version);
        window.dispatchEvent(new CustomEvent('fraud-sentinel-model-version'));
      }
      scoreMsg = riskExplanation(result);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      scoreMsg = '';
    } finally {
      loading = false;
    }
  }

  // ── Pipeline steps ─────────────────────────────────────────────────
  const steps = [
    { title: 'Score',   body: 'PyTorch classifier + autoencoder produce risk and anomaly signals.' },
    { title: 'Route',   body: 'Low → audit record only. Uncertain / High → review case opened.' },
    { title: 'Prepare', body: 'LangGraph agent retrieves policy context, generates grounded brief.' },
    { title: 'Decide',  body: 'Human analyst records rationale and action. Audit trail persists.' },
  ];

  const bands = [
    { band: 'high',      desc: '> 0.75 risk score — review case opened' },
    { band: 'uncertain', desc: '0.45–0.75 — review case opened' },
    { band: 'low',       desc: '< 0.45 — audit record only' },
  ];
</script>

<div class="grid-score">
  <!-- ── Left column ───────────────────────────────────────────────── -->
  <div class="stack">
    <div>
      <h1 class="page-title">Transaction Scoring</h1>
      <p class="page-sub">
        Score a single transaction against the PyTorch fraud classifier and autoencoder anomaly
        model. High or uncertain results open a review case.
      </p>
    </div>

    <!-- Quick actions -->
    <div class="row">
      <button class="btn" on:click={() => loadSample('ordinary')}>Load ordinary sample</button>
      <button class="btn" on:click={() => loadSample('fraud')}>Load fraud-like sample</button>
    </div>

    <!-- Score message / notice -->
    {#if scoreMsg && !result}
      <p class="notice">{scoreMsg}</p>
    {/if}

    <!-- Result panel -->
    {#if result}
      {@const color = riskColor(result.risk_band)}
      {@const borderColor = riskBorderColor(result.risk_band)}
      <div
        class="card"
        style="border-left: 4px solid {color}; border-color: {borderColor};"
      >
        <div class="row-between" style="margin-bottom: 16px; align-items: center;">
          <span style="font-weight: 600; font-size: 14px;">Prediction Result</span>
          <RiskPill band={result.risk_band} />
        </div>

        <div class="grid-2" style="margin-bottom: 16px;">
          <ScoreBar value={result.risk_score}    band={result.risk_band} label="Supervised fraud score" />
          <ScoreBar value={result.anomaly_score} band={result.risk_band} label="Autoencoder anomaly score" />
        </div>

        <p class="muted" style="font-size: 12px; margin-bottom: {result.case_id ? '12px' : 0};">{scoreMsg}</p>

        {#if result.case_id}
          <a href="/cases/{result.case_id}" class="btn btn-primary" style="width: fit-content;">
            Open review case →
          </a>
        {/if}

        <p class="muted" style="font-size: 11px; margin-top: 12px;">
          Model version: {result.model_version}
        </p>
      </div>
    {/if}

    {#if error}
      <div class="notice-error">{error}</div>
    {/if}

    <!-- Transaction form -->
    <form class="card stack" on:submit|preventDefault={submit}>
      <!-- Time + Amount -->
      <div class="grid-2">
        <label class="field-label">
          <span>Time</span>
          <input type="number" step="any" bind:value={transaction.Time} />
        </label>
        <label class="field-label">
          <span>Amount</span>
          <input type="number" step="any" bind:value={transaction.Amount} />
        </label>
      </div>

      <!-- V1–V28 toggle -->
      <button
        type="button"
        class="btn btn-ghost"
        style="width: 100%; justify-content: space-between;"
        on:click={() => (showPCA = !showPCA)}
        aria-expanded={showPCA}
      >
        <span>
          Model features V1–V28
          <span class="muted" style="font-size: 11px;">(anonymized PCA)</span>
        </span>
        <span
          style="font-size: 14px; display: inline-block; transform: rotate({showPCA ? 180 : 0}deg); transition: transform .2s;"
          aria-hidden="true"
        >▾</span>
      </button>

      {#if showPCA}
        <div class="grid-pca">
          {#each PCA_FIELDS as field}
            <label class="field-label" style="gap: 4px;">
              <span style="font-size: 10px; font-family: var(--font-mono);">{field}</span>
              <input
                type="number"
                step="any"
                style="padding: 6px 8px; font-size: 12px;"
                bind:value={transaction[field as keyof TransactionInput]}
              />
            </label>
          {/each}
        </div>
      {/if}

      <!-- Submit -->
      <div>
        <button type="submit" class="btn btn-primary" disabled={loading}>
          {#if loading}<span class="spinner"></span>{/if}
          {loading ? 'Scoring...' : 'Score transaction'}
        </button>
      </div>
    </form>
  </div>

  <!-- ── Right column ──────────────────────────────────────────────── -->
  <div class="stack">
    <!-- Pipeline -->
    <div class="card stack">
      <SectionLabel>Review pipeline</SectionLabel>
      {#each steps as step, i}
        <div style="display: flex; gap: 12px;">
          <div
            style="
              width: 22px; height: 22px; border-radius: 50%;
              background: var(--surface); border: 1px solid var(--accent);
              display: flex; align-items: center; justify-content: center;
              font-size: 10px; font-weight: 700; color: var(--accent); flex-shrink: 0; margin-top: 1px;
            "
          >{i + 1}</div>
          <div>
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 2px;">{step.title}</div>
            <div style="font-size: 11px; color: var(--muted); line-height: 1.5;">{step.body}</div>
          </div>
        </div>
      {/each}
    </div>

    <!-- Risk bands -->
    <div class="card stack">
      <SectionLabel>Risk bands</SectionLabel>
      {#each bands as b}
        <div style="display: flex; gap: 10px; align-items: flex-start;">
          <RiskPill band={b.band} />
          <span style="font-size: 11px; color: var(--muted); line-height: 1.4; padding-top: 2px;">{b.desc}</span>
        </div>
      {/each}
    </div>

    <!-- Field reference -->
    <div class="card">
      <SectionLabel>Field reference</SectionLabel>
      <div style="font-size: 11px; color: var(--muted); line-height: 1.8;">
        <div><span class="mono" style="color: var(--text);">Time</span> — seconds since dataset start</div>
        <div><span class="mono" style="color: var(--text);">Amount</span> — transaction amount</div>
        <div><span class="mono" style="color: var(--text);">V1–V28</span> — anonymized PCA features, not business-readable</div>
      </div>
      <div
        style="
          margin-top: 10px; padding: 8px 10px; background: var(--surface);
          border-radius: var(--radius-sm); border-left: 3px solid var(--uncertain);
          font-size: 11px; color: var(--muted); line-height: 1.5;
        "
      >
        High scores mean <em>similar to fraud patterns in training data</em> — not confirmed fraud.
      </div>
    </div>
  </div>
</div>
