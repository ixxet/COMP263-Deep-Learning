<script lang="ts">
  import { predict, uploadBatch } from '$lib/api';
  import type { PredictionResponse, TransactionInput } from '$lib/types';

  const pcaFields = Array.from({ length: 28 }, (_, index) => `V${index + 1}`);

  let transaction: TransactionInput = {
    Time: 0,
    Amount: 129.5,
    ...Object.fromEntries(pcaFields.map((field) => [field, 0]))
  } as TransactionInput;
  let result: PredictionResponse | null = null;
  let error = '';
  let loading = false;
  let batchMessage = '';

  function loadSuspiciousSample() {
    transaction = {
      Time: 406,
      Amount: 529,
      V1: -2.31,
      V2: 1.95,
      V3: -1.61,
      V4: 3.99,
      V5: -0.52,
      V6: -1.43,
      V7: -2.54,
      V8: 1.39,
      V9: -2.77,
      V10: -2.77,
      V11: 3.2,
      V12: -2.9,
      V13: -0.59,
      V14: -4.29,
      V15: 0.39,
      V16: -1.14,
      V17: -2.83,
      V18: -0.02,
      V19: 0.42,
      V20: 0.13,
      V21: 0.52,
      V22: -0.04,
      V23: -0.47,
      V24: 0.32,
      V25: 0.04,
      V26: 0.18,
      V27: 0.26,
      V28: -0.14
    } as TransactionInput;
  }

  async function submitPrediction() {
    loading = true;
    error = '';
    result = null;
    try {
      result = await predict(transaction);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function submitBatch(event: Event) {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) return;
    batchMessage = '';
    error = '';
    try {
      const response = await uploadBatch(file);
      batchMessage = `${response.accepted_rows} accepted, ${response.rejected_rows} rejected, ${response.case_ids.length} cases opened.`;
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      input.value = '';
    }
  }
</script>

<div class="grid">
  <section class="band">
    <h1 class="section-title">Transaction Scoring</h1>
    <p class="muted">
      Submit one Kaggle-format transaction or upload a CSV with Time, Amount, and V1 through V28.
    </p>

    <div class="metric-row" aria-label="Pipeline status">
      <div class="metric"><strong>30</strong><span>input features</span></div>
      <div class="metric"><strong>2</strong><span>model signals</span></div>
      <div class="metric"><strong>3</strong><span>risk bands</span></div>
    </div>

    <div class="actions">
      <button class="secondary" type="button" on:click={loadSuspiciousSample}>Load review sample</button>
      <label>
        Batch CSV
        <input type="file" accept=".csv,text/csv" on:change={submitBatch} />
      </label>
    </div>

    {#if batchMessage}
      <p class="muted">{batchMessage}</p>
    {/if}

    <form on:submit|preventDefault={submitPrediction}>
      <div class="form-grid">
        <label>
          Time
          <input type="number" step="any" bind:value={transaction.Time} />
        </label>
        <label>
          Amount
          <input type="number" step="any" bind:value={transaction.Amount} />
        </label>
        {#each pcaFields as field}
          <label>
            {field}
            <input type="number" step="any" bind:value={transaction[field as keyof TransactionInput]} />
          </label>
        {/each}
      </div>
      <div class="actions">
        <button disabled={loading}>{loading ? 'Scoring...' : 'Score transaction'}</button>
      </div>
    </form>

    {#if error}
      <section class="band result high">
        <strong>Request failed</strong>
        <p>{error}</p>
      </section>
    {/if}

    {#if result}
      <section class={`band result ${result.risk_band}`}>
        <h2 class="section-title">Prediction Result</h2>
        <div class="metric-row">
          <div class="metric">
            <strong>{(result.risk_score * 100).toFixed(1)}%</strong>
            <span>risk score</span>
          </div>
          <div class="metric">
            <strong>{(result.anomaly_score * 100).toFixed(1)}%</strong>
            <span>anomaly score</span>
          </div>
          <div class="metric">
            <strong><span class={`pill ${result.risk_band}`}>{result.risk_band}</span></strong>
            <span>risk band</span>
          </div>
        </div>
        <p class="muted">Model version: {result.model_version}</p>
        {#if result.case_id}
          <p><a href={`/cases/${result.case_id}`}>Open case {result.case_id}</a></p>
        {/if}
      </section>
    {/if}
  </section>

  <aside class="band">
    <img
      class="ops-image"
      src="https://images.unsplash.com/photo-1563013544-824ae1b704d3?auto=format&fit=crop&w=900&q=80"
      alt="Payment terminal with credit card"
    />
    <h2 class="section-title">Review Gates</h2>
    <p class="muted">
      Low-risk transactions are stored for audit. Uncertain and high-risk predictions open cases for
      LangGraph review, policy retrieval, grounded briefing, and analyst action.
    </p>
    <ul>
      <li>Model score drives the risk band.</li>
      <li>RAG explains policy and limits.</li>
      <li>Humans approve, escalate, or dismiss.</li>
    </ul>
  </aside>
</div>

