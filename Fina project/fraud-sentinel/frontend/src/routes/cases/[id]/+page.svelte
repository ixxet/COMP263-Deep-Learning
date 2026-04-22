<script lang="ts">
  import { getCase, reviewCase } from '$lib/api';
  import type { CaseDetail } from '$lib/types';
  import { page } from '$app/stores';

  let item: CaseDetail | null = null;
  let error = '';
  let loading = true;
  let reviewer = 'analyst';
  let rationale = '';
  const actionCopy = {
    approve: 'Confirm the model flag and close the case as reviewed.',
    escalate: 'Send the case to deeper investigation or a real fraud operations queue.',
    dismiss: 'Close this as a false alarm or not enough evidence.'
  };

  async function load() {
    loading = true;
    error = '';
    try {
      const caseId = $page.params.id;
      if (!caseId) throw new Error('case id is missing');
      item = await getCase(caseId);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  async function submit(action: 'approve' | 'escalate' | 'dismiss') {
    if (!item) return;
    error = '';
    try {
      await reviewCase(item.case_id, { action, reviewer, rationale });
      rationale = '';
      await load();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    }
  }

  function pct(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  load();
</script>

{#if loading}
  <section class="band"><p class="muted">Loading case...</p></section>
{:else if error}
  <section class="band result high"><p>{error}</p></section>
{:else if item}
  <div class="grid">
    <section class="band">
      <h1 class="section-title">Case {item.case_id}</h1>
      <p class="muted">
        This page is the analyst gate. The model opened the case; the human decision records what
        should happen next.
      </p>
      <div class="metric-row">
        <div class="metric">
          <strong><span class={`pill ${item.risk_band}`}>{item.risk_band}</span></strong>
          <span>risk band</span>
        </div>
        <div class="metric">
          <strong>{pct(item.risk_score)}</strong>
          <span>supervised fraud score</span>
          <div class="score-bar"><span style={`width: ${pct(item.risk_score)}`}></span></div>
        </div>
        <div class="metric">
          <strong>{pct(item.anomaly_score)}</strong>
          <span>autoencoder anomaly score</span>
          <div class="score-bar"><span style={`width: ${pct(item.anomaly_score)}`}></span></div>
        </div>
      </div>

      <h2 class="section-title">Agent Brief</h2>
      <p>{item.brief ?? 'The agent worker has not produced a brief yet. The case can still be reviewed from the model scores and transaction payload.'}</p>

      <h2 class="section-title">Decision Meaning</h2>
      <div class="step-list">
        {#each Object.entries(actionCopy) as [action, description]}
          <p><strong>{action}</strong><span>{description}</span></p>
        {/each}
      </div>

      <h2 class="section-title">Human Review</h2>
      <label>
        Reviewer
        <input bind:value={reviewer} />
      </label>
      <label>
        Rationale
        <textarea bind:value={rationale} placeholder="Record the analyst reason for this action"></textarea>
      </label>
      <div class="actions">
        <button class="secondary" disabled={!rationale} on:click={() => submit('approve')}>Approve flag</button>
        <button class="warning" disabled={!rationale} on:click={() => submit('escalate')}>Escalate investigation</button>
        <button class="danger" disabled={!rationale} on:click={() => submit('dismiss')}>Dismiss false alarm</button>
      </div>

      {#if item.reviews.length > 0}
        <h2 class="section-title">Review History</h2>
        <div class="timeline">
          {#each item.reviews as review}
            <p>
              <strong>{String(review.action)}</strong>
              <span>{String(review.reviewer)}: {String(review.rationale)}</span>
            </p>
          {/each}
        </div>
      {/if}
    </section>

    <aside class="band">
      <h2 class="section-title">Status</h2>
      <p><span class={`pill ${item.status}`}>{item.status}</span></p>
      <p class="muted">
        Pending cases need a rationale and one review action. Reviewed cases stay in the queue for
        audit history.
      </p>

      <h2 class="section-title">Transaction</h2>
      <table class="table">
        <tbody>
          {#each Object.entries(item.transaction) as [key, value]}
            <tr><th>{key}</th><td>{Number(value).toFixed(4)}</td></tr>
          {/each}
        </tbody>
      </table>

      <h2 class="section-title">Policy Context</h2>
      {#if item.policy_context.length === 0}
        <p class="muted">No policy context saved yet.</p>
      {:else}
        {#each item.policy_context as context, index}
          <p>
            <strong>[{index + 1}] {String(context.title ?? 'Policy')}</strong><br />
            <span class="muted">{String(context.source ?? 'local policy corpus')}</span>
          </p>
        {/each}
      {/if}
    </aside>
  </div>
{/if}
