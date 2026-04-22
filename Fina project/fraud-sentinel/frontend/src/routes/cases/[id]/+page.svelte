<script lang="ts">
  import { getCase, reviewCase } from '$lib/api';
  import type { CaseDetail } from '$lib/types';
  import { page } from '$app/stores';

  let item: CaseDetail | null = null;
  let error = '';
  let loading = true;
  let reviewer = 'analyst';
  let rationale = '';

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
      <div class="metric-row">
        <div class="metric">
          <strong><span class={`pill ${item.risk_band}`}>{item.risk_band}</span></strong>
          <span>risk band</span>
        </div>
        <div class="metric"><strong>{(item.risk_score * 100).toFixed(1)}%</strong><span>risk</span></div>
        <div class="metric"><strong>{(item.anomaly_score * 100).toFixed(1)}%</strong><span>anomaly</span></div>
      </div>

      <h2 class="section-title">Agent Brief</h2>
      <p>{item.brief ?? 'The agent worker has not produced a brief yet.'}</p>

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
        <button class="secondary" disabled={!rationale} on:click={() => submit('approve')}>Approve</button>
        <button class="warning" disabled={!rationale} on:click={() => submit('escalate')}>Escalate</button>
        <button class="danger" disabled={!rationale} on:click={() => submit('dismiss')}>Dismiss</button>
      </div>
    </section>

    <aside class="band">
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
          <p><strong>[{index + 1}] {String(context.title ?? 'Policy')}</strong></p>
        {/each}
      {/if}
    </aside>
  </div>
{/if}
