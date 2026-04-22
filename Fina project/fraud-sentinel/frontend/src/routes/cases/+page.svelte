<script lang="ts">
  import { listCases } from '$lib/api';
  import type { CaseSummary } from '$lib/types';

  let cases: CaseSummary[] = [];
  let error = '';
  let loading = true;

  async function load() {
    loading = true;
    error = '';
    try {
      cases = await listCases();
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $: pending = cases.filter((item) =>
    ['pending_review', 'awaiting_human_review'].includes(item.status)
  ).length;
  $: high = cases.filter((item) => item.risk_band === 'high').length;

  load();
</script>

<section class="band">
  <h1 class="section-title">Case Queue</h1>
  <div class="metric-row">
    <div class="metric"><strong>{cases.length}</strong><span>total cases</span></div>
    <div class="metric"><strong>{pending}</strong><span>pending review</span></div>
    <div class="metric"><strong>{high}</strong><span>high risk</span></div>
  </div>
  <div class="actions">
    <button class="secondary" on:click={load}>Refresh</button>
  </div>

  {#if loading}
    <p class="muted">Loading cases...</p>
  {:else if error}
    <p>{error}</p>
  {:else if cases.length === 0}
    <p class="muted">No cases yet. Score a high or uncertain transaction to open one.</p>
  {:else}
    <table class="table">
      <thead>
        <tr>
          <th>Case</th>
          <th>Status</th>
          <th>Risk</th>
          <th>Scores</th>
          <th>Created</th>
        </tr>
      </thead>
      <tbody>
        {#each cases as item}
          <tr>
            <td><a href={`/cases/${item.case_id}`}>{item.case_id}</a></td>
            <td><span class={`pill ${item.status}`}>{item.status}</span></td>
            <td><span class={`pill ${item.risk_band}`}>{item.risk_band}</span></td>
            <td>
              risk {(item.risk_score * 100).toFixed(1)}%<br />
              anomaly {(item.anomaly_score * 100).toFixed(1)}%
            </td>
            <td>{item.created_at ? new Date(item.created_at).toLocaleString() : 'n/a'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>
