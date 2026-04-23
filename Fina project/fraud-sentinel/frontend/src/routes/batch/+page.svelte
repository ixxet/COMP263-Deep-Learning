<script lang="ts">
  import { uploadBatch } from '$lib/api';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import Stat from '$lib/components/Stat.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';

  // ── Types ──────────────────────────────────────────────────────────
  type BatchResponse = {
    accepted_rows: number;
    rejected_rows: number;
    prediction_ids: string[];
    case_ids: string[];
  };

  type ScoredRow = {
    row_index: number;
    risk_score: number;
    anomaly_score: number;
    risk_band: string;
    prediction_id: string;
    case_id: string | null;
  };

  type RejectedRow = {
    row_index: number;
    reason: string;
  };

  // ── State ──────────────────────────────────────────────────────────
  let state: 'idle' | 'loading' | 'done' = 'idle';
  let error = '';
  let raw: BatchResponse | null = null;

  // The real API returns accepted_rows, rejected_rows, prediction_ids, case_ids.
  // The UI also surfaces per-row detail when available.
  // If the backend returns extended row data, map it here; otherwise derive from top-level counts.
  let scoredRows: ScoredRow[] = [];
  let rejectedRows: RejectedRow[] = [];

  let fileInput: HTMLInputElement;

  // ── Upload ─────────────────────────────────────────────────────────
  async function uploadFile(file: File) {
    state = 'loading'; error = '';

    try {
      raw = await uploadBatch(file);

      // If the API returns extended per-row data in the future, map it here.
      // For now, derive summary rows from what the API does return.
      scoredRows  = [];
      rejectedRows = [];

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

  function reset() {
    state = 'idle';
    raw = null;
    scoredRows = [];
    rejectedRows = [];
    error = '';
  }

  // ── Helpers ────────────────────────────────────────────────────────
  function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

  $: casesOpened = raw?.case_ids.length ?? 0;
  $: auditOnly   = (raw?.accepted_rows ?? 0) - casesOpened;

  const RISK_COLOR: Record<string, string> = {
    high:      'var(--high)',
    uncertain: 'var(--uncertain)',
    low:       'var(--accent)',
  };
</script>

<div class="page">
  <h1 class="page-title">Batch Upload</h1>
  <p class="page-sub">
    Upload a CSV with Kaggle credit-card fraud schema: Time, Amount, V1–V28, optional Class.
    Up to 500 rows per batch. Each row is scored and persisted independently.
  </p>

  <!-- ── Idle: drop zone ─────────────────────────────────────────── -->
  {#if state === 'idle'}
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
      <div class="muted" style="font-size: 12px;">Accepts .csv — max 500 rows</div>
      <input
        bind:this={fileInput}
        type="file"
        accept=".csv,text/csv"
        style="display: none;"
        on:change={handleFile}
      />
    </div>

    {#if error}
      <div class="notice-error" style="margin-top: 16px;">{error}</div>
    {/if}

  <!-- ── Loading ─────────────────────────────────────────────────── -->
  {:else if state === 'loading'}
    <div class="card" style="text-align: center; padding: 48px;">
      <div class="spinner-lg"></div>
      <p class="muted" style="font-size: 13px;">Parsing and scoring rows...</p>
    </div>

  <!-- ── Done ───────────────────────────────────────────────────── -->
  {:else if state === 'done' && raw}
    <div class="stack">

      <!-- Stats row -->
      <div class="grid-4">
        <Stat label="Accepted rows" value={raw.accepted_rows} accent="var(--accent)" />
        <Stat label="Rejected rows" value={raw.rejected_rows} accent={raw.rejected_rows > 0 ? 'var(--high)' : 'var(--muted)'} />
        <Stat label="Cases opened"  value={casesOpened}       accent={casesOpened > 0 ? 'var(--uncertain)' : 'var(--muted)'} />
        <Stat label="Audit-only"    value={auditOnly}          accent="var(--muted)" />
      </div>

      <!-- Case links -->
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
                <span class="muted" style="font-size: 11px;">Review →</span>
              </a>
            {/each}
          </div>
        </div>
      {/if}

      <!-- Per-row detail (rendered when API returns extended data) -->
      {#if scoredRows.length > 0}
        <div class="card" style="padding: 0; overflow: hidden;">
          <div style="display: flex; justify-content: space-between; align-items: center; padding: 16px 24px; border-bottom: 1px solid var(--border);">
            <div class="section-label" style="margin: 0;">Scored rows (preview)</div>
            <button class="btn" style="font-size: 11px; padding: 6px 12px;">↓ Download CSV</button>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>Row</th>
                <th>Risk score</th>
                <th>Anomaly score</th>
                <th>Band</th>
                <th>Case</th>
              </tr>
            </thead>
            <tbody>
              {#each scoredRows as row}
                <tr>
                  <td><span class="mono muted" style="font-size: 12px;">{row.row_index}</span></td>
                  <td><span class="mono" style="font-size: 12px;">{pct(row.risk_score)}</span></td>
                  <td><span class="mono" style="font-size: 12px;">{pct(row.anomaly_score)}</span></td>
                  <td><RiskPill band={row.risk_band} /></td>
                  <td>
                    {#if row.case_id}
                      <a href="/cases/{row.case_id}" class="mono link-accent" style="font-size: 11px;">{row.case_id.slice(0, 8)}</a>
                    {:else}
                      <span class="muted">—</span>
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      {/if}

      <!-- Rejected rows (rendered when API returns extended data) -->
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

      <!-- Totals notice when per-row data is unavailable -->
      {#if scoredRows.length === 0 && rejectedRows.length === 0}
        <div class="notice">
          {raw.accepted_rows} rows accepted and scored.
          {raw.rejected_rows} rows rejected.
          {raw.case_ids.length} review {raw.case_ids.length === 1 ? 'case' : 'cases'} opened.
          Per-row detail is available when the API returns extended batch results.
        </div>
      {/if}

      <button class="btn" style="align-self: flex-start;" on:click={reset}>
        Upload another batch
      </button>
    </div>
  {/if}
</div>
