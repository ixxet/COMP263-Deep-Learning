<script lang="ts">
  import { predict, uploadBatch } from '$lib/api';
  import type { PredictionResponse, TransactionInput } from '$lib/types';

  const pcaFields = Array.from({ length: 28 }, (_, index) => `V${index + 1}`);
  const sampleRows: Array<TransactionInput & { Class: number; note: string }> = [
    {
      Time: 0,
      Amount: 43.75,
      ...Object.fromEntries(pcaFields.map((field) => [field, 0])),
      Class: 0,
      note: 'ordinary-looking baseline row'
    } as TransactionInput & { Class: number; note: string },
    {
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
      V28: -0.14,
      Class: 1,
      note: 'known fraud-like review row'
    } as TransactionInput & { Class: number; note: string },
    {
      Time: 1200,
      Amount: 2400,
      ...Object.fromEntries(pcaFields.map((field) => [field, 0])),
      V10: -2.2,
      V14: -2.8,
      V17: -1.8,
      Class: 0,
      note: 'stress-test row'
    } as TransactionInput & { Class: number; note: string }
  ];

  let transaction: TransactionInput = {
    Time: 0,
    Amount: 129.5,
    ...Object.fromEntries(pcaFields.map((field) => [field, 0]))
  } as TransactionInput;
  let result: PredictionResponse | null = null;
  let error = '';
  let loading = false;
  let batchMessage = '';
  let scoreMessage = '';

  function loadSuspiciousSample() {
    loadSample(1);
  }

  function loadOrdinarySample() {
    loadSample(0);
  }

  function loadSample(index: number) {
    const { Class: _class, note: _note, ...sample } = sampleRows[index];
    transaction = { ...sample };
    result = null;
    scoreMessage = `Loaded ${sampleRows[index].note}. Press Score transaction to run the model.`;
  }

  function downloadSampleCsv() {
    const columns = ['Time', ...pcaFields, 'Amount', 'Class', 'note'];
    const rows = sampleRows.map((row) =>
      columns
        .map((column) => JSON.stringify(String(row[column as keyof typeof row] ?? '')))
        .join(',')
    );
    const csv = `${columns.join(',')}\n${rows.join('\n')}\n`;
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'fraud-sentinel-sample-transactions.csv';
    link.click();
    URL.revokeObjectURL(url);
  }

  function formatPercent(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function riskExplanation(prediction: PredictionResponse) {
    if (prediction.risk_band === 'high') {
      return 'A high-risk case was opened because the supervised fraud score or anomaly signal crossed the review gate.';
    }
    if (prediction.risk_band === 'uncertain') {
      return 'An uncertain case was opened because one signal is elevated enough for analyst review.';
    }
    return 'This was stored for audit only. No analyst case was opened.';
  }

  async function submitPrediction() {
    loading = true;
    error = '';
    result = null;
    scoreMessage = 'Scoring transaction...';
    try {
      result = await predict(transaction);
      scoreMessage = riskExplanation(result);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
      scoreMessage = '';
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
      Score one transaction or upload a CSV. The model expects Kaggle credit-card fraud fields:
      Time, Amount, and anonymized PCA features V1 through V28. A CSV may also include Class; it is
      ignored during inference.
    </p>

    <div class="metric-row" aria-label="Pipeline status">
      <div class="metric"><strong>30</strong><span>input features</span></div>
      <div class="metric"><strong>2</strong><span>model signals</span></div>
      <div class="metric"><strong>3</strong><span>risk bands</span></div>
    </div>

    <div class="actions">
      <button class="secondary" type="button" on:click={loadOrdinarySample}>Load ordinary sample</button>
      <button class="secondary" type="button" on:click={loadSuspiciousSample}>Load review sample</button>
      <button class="secondary" type="button" on:click={downloadSampleCsv}>Download sample CSV</button>
      <label>
        Batch CSV
        <input type="file" accept=".csv,text/csv" on:change={submitBatch} />
      </label>
    </div>

    {#if scoreMessage}
      <p class="notice">{scoreMessage}</p>
    {/if}

    {#if batchMessage}
      <p class="notice">{batchMessage}</p>
    {/if}

    {#if result}
      <section class={`result-panel ${result.risk_band}`} aria-live="polite">
        <div>
          <h2 class="section-title">Prediction Result</h2>
          <p class="muted">
            The model score ranks how similar this transaction is to fraud patterns in the training
            data. It is not a payment decision and the LLM does not decide fraud.
          </p>
        </div>
        <div class="metric-row">
          <div class="metric">
            <strong>{formatPercent(result.risk_score)}</strong>
            <span>supervised fraud score</span>
            <div class="score-bar"><span style={`width: ${formatPercent(result.risk_score)}`}></span></div>
          </div>
          <div class="metric">
            <strong>{formatPercent(result.anomaly_score)}</strong>
            <span>autoencoder anomaly score</span>
            <div class="score-bar"><span style={`width: ${formatPercent(result.anomaly_score)}`}></span></div>
          </div>
          <div class="metric">
            <strong><span class={`pill ${result.risk_band}`}>{result.risk_band}</span></strong>
            <span>review route</span>
            <p class="metric-note">{result.case_id ? 'Case opened for review.' : 'Audit record only.'}</p>
          </div>
        </div>
        <p class="muted">Model version: {result.model_version}</p>
        {#if result.case_id}
          <p><a class="callout-link" href={`/cases/${result.case_id}`}>Open review case</a></p>
        {/if}
      </section>
    {/if}

    <form on:submit|preventDefault={submitPrediction}>
      <div class="field-help">
        <strong>What are these fields?</strong>
        <span>
          Time is seconds from the first transaction in the dataset. Amount is the transaction
          amount. V1-V28 are anonymized PCA features, so they do not map back to merchant, card, or
          customer names.
        </span>
      </div>
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
        <button type="submit" disabled={loading}>{loading ? 'Scoring...' : 'Score transaction'}</button>
      </div>
    </form>

    {#if error}
      <section class="band result high">
        <strong>Request failed</strong>
        <p>{error}</p>
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
    <div class="step-list">
      <p><strong>1. Score</strong><span>The PyTorch model produces fraud and anomaly signals.</span></p>
      <p><strong>2. Route</strong><span>Low risk closes to audit; uncertain and high open cases.</span></p>
      <p><strong>3. Review</strong><span>LangGraph drafts context, then a human records the decision.</span></p>
    </div>
    <h2 class="section-title">CSV Source</h2>
    <p class="muted">
      Use the sample CSV here, or use rows from Kaggle creditcard.csv. For batch scoring, keep Time,
      Amount, V1-V28, and optional Class.
    </p>
  </aside>
</div>
