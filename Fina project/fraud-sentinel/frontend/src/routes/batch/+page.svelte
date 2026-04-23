<script lang="ts">
  import { uploadBatch } from '$lib/api';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import Stat from '$lib/components/Stat.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';
  import type { BatchPredictionResponse, BatchPredictionRow, BatchRejectedRow, RiskBand } from '$lib/types';

  let state: 'idle' | 'loading' | 'done' = 'idle';
  let error = '';
  let raw: BatchPredictionResponse | null = null;
  let scoredRows: BatchPredictionRow[] = [];
  let rejectedRows: BatchRejectedRow[] = [];
  let downloadingSample = false;
  let fileInput: HTMLInputElement;

  async function uploadFile(file: File) {
    state = 'loading';
    error = '';
    try {
      raw = await uploadBatch(file);
      scoredRows = raw.rows ?? [];
      rejectedRows = raw.rejections ?? [];
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      state = 'idle';
    } finally {
      if (state === 'loading') state = 'done';
    }
  }

  async function handleFile(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;

    try {
      await uploadFile(file);
    } finally {
      input.value = '';
    }
  }

  function handleDrop(event: DragEvent) {
    event.preventDefault();
    const file = event.dataTransfer?.files?.[0];
    if (file) void uploadFile(file);
  }

  async function downloadDemoCsv() {
    downloadingSample = true;
    error = '';
    try {
      const response = await fetch('/api/v1/samples/demo.csv');
      if (!response.ok) throw new Error(await response.text());
      saveBlob('fraud-sentinel-demo-transactions.csv', await response.blob());
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      downloadingSample = false;
    }
  }

  function downloadResultsCsv() {
    const columns = ['row_index', 'prediction_id', 'risk_score', 'anomaly_score', 'risk_band', 'model_version', 'case_id'];
    const lines = [
      columns.join(','),
      ...scoredRows.map((row) => columns.map((column) => csvEscape(row[column as keyof BatchPredictionRow])).join(','))
    ];
    saveTextFile('fraud-sentinel-batch-results.csv', `${lines.join('\n')}\n`);
  }

  function reset() {
    state = 'idle';
    raw = null;
    scoredRows = [];
    rejectedRows = [];
    error = '';
  }

  function saveBlob(fileName: string, blob: Blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = fileName;
    link.click();
    URL.revokeObjectURL(url);
  }

  function saveTextFile(fileName: string, content: string) {
    saveBlob(fileName, new Blob([content], { type: 'text/csv;charset=utf-8' }));
  }

  function csvEscape(value: unknown) {
    const text = String(value ?? '');
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  }

  function pct(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  $: casesOpened = raw?.case_ids.length ?? 0;
  $: auditOnly = (raw?.accepted_rows ?? 0) - casesOpened;
  $: riskMix = {
    low: scoredRows.filter((row) => row.risk_band === 'low').length,
    uncertain: scoredRows.filter((row) => row.risk_band === 'uncertain').length,
    high: scoredRows.filter((row) => row.risk_band === 'high').length
  };
  $: maxRisk = Math.max(1, ...Object.values(riskMix));

  const RISK_COLOR: Record<RiskBand, string> = {
    high: 'var(--high)',
    uncertain: 'var(--uncertain)',
    low: 'var(--accent)'
  };

  const SAMPLE_ROWS = [
    { band: 'low' as const, label: 'Ordinary', body: 'Baseline rows stay audit-only.' },
    { band: 'uncertain' as const, label: 'Uncertain', body: 'Borderline rows open review cases.' },
    { band: 'high' as const, label: 'Fraud-like', body: 'Strong pattern matches open review cases.' }
  ];
</script>

<div class="page">
  <h1 class="page-title">Batch Upload</h1>
  <p class="page-sub">
    Upload Kaggle-format CSV rows. Each accepted row becomes a prediction record; uncertain and
    high-risk rows also open review cases.
  </p>

  {#if state === 'idle'}
    <div class="grid-2" style="align-items: stretch;">
      <!-- svelte-ignore a11y-click-events-have-key-events -->
      <!-- svelte-ignore a11y-no-static-element-interactions -->
      <div
        class="drop-zone"
        on:click={() => fileInput.click()}
        on:dragover|preventDefault
        on:drop={handleDrop}
      >
        <div style="font-size: 32px; margin-bottom: 12px;" aria-hidden="true">⊞</div>
        <div style="font-weight: 600; margin-bottom: 6px;">Drop CSV file or click to browse</div>
        <div class="muted" style="font-size: 12px;">Time, Amount, V1-V28, optional Class. Max 500 rows.</div>
        <input
          bind:this={fileInput}
          type="file"
          accept=".csv,text/csv"
          style="display: none;"
          on:change={handleFile}
        />
      </div>

      <div class="card stack">
        <SectionLabel>Demo CSV</SectionLabel>
        <p class="muted" style="font-size: 12px; line-height: 1.6;">
          Download a model-aware sample with ordinary, uncertain, and fraud-like rows. The optional
          Class column is accepted for Kaggle compatibility; runtime scoring ignores it.
        </p>
        <div class="stack" style="gap: 8px;">
          {#each SAMPLE_ROWS as sample}
            <div style="display: flex; align-items: flex-start; gap: 10px;">
              <RiskPill band={sample.band} />
              <div>
                <div style="font-size: 12px; font-weight: 600;">{sample.label}</div>
                <div class="muted" style="font-size: 11px;">{sample.body}</div>
              </div>
            </div>
          {/each}
        </div>
        <button class="btn btn-primary" style="width: fit-content;" on:click={downloadDemoCsv} disabled={downloadingSample}>
          {#if downloadingSample}<span class="spinner"></span>{/if}
          {downloadingSample ? 'Preparing CSV...' : 'Download demo CSV'}
        </button>
      </div>
    </div>

    {#if error}
      <div class="notice-error" style="margin-top: 16px;">{error}</div>
    {/if}

  {:else if state === 'loading'}
    <div class="card" style="text-align: center; padding: 48px;">
      <div class="spinner-lg"></div>
      <p class="muted" style="font-size: 13px;">Parsing and scoring rows...</p>
    </div>

  {:else if state === 'done' && raw}
    <div class="stack">
      <div class="grid-4">
        <Stat label="Accepted rows" value={raw.accepted_rows} accent="var(--accent)" />
        <Stat label="Rejected rows" value={raw.rejected_rows} accent={raw.rejected_rows > 0 ? 'var(--high)' : 'var(--muted)'} />
        <Stat label="Cases opened" value={casesOpened} accent={casesOpened > 0 ? 'var(--uncertain)' : 'var(--muted)'} />
        <Stat label="Audit-only" value={auditOnly} accent="var(--muted)" />
      </div>

      <div class="grid-2">
        <div class="card">
          <SectionLabel>Risk distribution</SectionLabel>
          {#each Object.entries(riskMix) as [band, count]}
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
              <div style="width: 90px; flex-shrink: 0;">
                <RiskPill band={band} />
              </div>
              <div style="flex: 1; height: 8px; background: rgba(255,255,255,.05); border-radius: 4px; overflow: hidden;">
                <div
                  style="
                    width: {(count / maxRisk) * 100}%;
                    height: 100%;
                    background: {RISK_COLOR[band as RiskBand]};
                    border-radius: 4px;
                  "
                ></div>
              </div>
              <span class="mono" style="font-size: 13px; font-weight: 700; min-width: 24px; text-align: right;">{count}</span>
            </div>
          {/each}
        </div>

        <div class="card">
          <SectionLabel>Routing result</SectionLabel>
          <div class="stack" style="gap: 10px; font-size: 12px;">
            <div style="display: flex; justify-content: space-between; gap: 12px;">
              <span class="muted">Low-risk audit records</span>
              <a class="link-accent mono" href="/history?filter=audit-only">{auditOnly}</a>
            </div>
            <div style="display: flex; justify-content: space-between; gap: 12px;">
              <span class="muted">Review cases opened</span>
              <a class="link-accent mono" href="/cases">{casesOpened}</a>
            </div>
            <p class="muted" style="font-size: 11px; line-height: 1.6;">
              Case Queue only contains uncertain and high-risk predictions. All accepted rows,
              including audit-only lows, are visible in Prediction History.
            </p>
          </div>
        </div>
      </div>

      {#if raw.case_ids.length > 0}
        <div class="card">
          <SectionLabel>Opened cases</SectionLabel>
          <div class="stack" style="gap: 6px;">
            {#each raw.case_ids as caseId}
              <a
                href="/cases/{caseId}"
                style="display: flex; align-items: center; gap: 10px; padding: 8px 12px;
                  background: var(--surface); border-radius: var(--radius-md);
                  border-left: 3px solid var(--uncertain); text-decoration: none;"
              >
                <span class="mono" style="font-size: 12px; color: var(--accent);">{caseId.slice(0, 8)}</span>
                <span class="muted" style="font-size: 11px;">Review</span>
              </a>
            {/each}
          </div>
        </div>
      {/if}

      {#if scoredRows.length > 0}
        <div class="card" style="padding: 0; overflow: hidden;">
          <div style="display: flex; justify-content: space-between; align-items: center; gap: 12px; padding: 16px 24px; border-bottom: 1px solid var(--border);">
            <div>
              <div class="section-label" style="margin: 0;">Scored rows</div>
              <div class="muted" style="font-size: 11px; margin-top: 4px;">One prediction record per accepted CSV row.</div>
            </div>
            <button class="btn" style="font-size: 11px; padding: 6px 12px;" on:click={downloadResultsCsv}>
              Download results CSV
            </button>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Row</th>
                <th>Risk score</th>
                <th>Anomaly score</th>
                <th>Band</th>
                <th>Prediction</th>
                <th>Route</th>
              </tr>
            </thead>
            <tbody>
              {#each scoredRows as row}
                <tr>
                  <td><span class="mono muted" style="font-size: 12px;">{row.row_index}</span></td>
                  <td><span class="mono" style="font-size: 12px;">{pct(row.risk_score)}</span></td>
                  <td><span class="mono" style="font-size: 12px;">{pct(row.anomaly_score)}</span></td>
                  <td><RiskPill band={row.risk_band} /></td>
                  <td><span class="mono muted" style="font-size: 11px;">{row.prediction_id.slice(0, 8)}</span></td>
                  <td>
                    {#if row.case_id}
                      <a href="/cases/{row.case_id}" class="link-accent" style="font-size: 12px;">Review case</a>
                    {:else}
                      <span class="muted" style="font-size: 12px;">Audit-only</span>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      {#if rejectedRows.length > 0}
        <div class="card">
          <SectionLabel>Rejected rows</SectionLabel>
          <div class="stack" style="gap: 6px;">
            {#each rejectedRows as row}
              <div style="
                padding: 8px 12px; background: var(--surface);
                border-radius: var(--radius-md); border-left: 3px solid var(--high);
              ">
                <span class="mono muted" style="font-size: 11px;">Row {row.row_index}</span>
                <div style="font-size: 12px; margin-top: 2px;">{row.reason}</div>
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if scoredRows.length === 0 && rejectedRows.length === 0}
        <div class="notice">
          {raw.accepted_rows} rows accepted and scored. {raw.rejected_rows} rows rejected.
          This API version did not return row-level detail.
        </div>
      {/if}

      <div class="row">
        <button class="btn" on:click={reset}>Upload another batch</button>
        <a href="/history" class="btn btn-ghost">View prediction history</a>
      </div>
    </div>
  {/if}
</div>
