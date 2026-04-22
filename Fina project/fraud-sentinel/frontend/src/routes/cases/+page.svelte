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
  $: reviewed = cases.filter((item) =>
    ['approved', 'escalated', 'dismissed'].includes(item.status)
  ).length;
  $: riskCounts = {
    high: cases.filter((item) => item.risk_band === 'high').length,
    uncertain: cases.filter((item) => item.risk_band === 'uncertain').length,
    low: cases.filter((item) => item.risk_band === 'low').length
  };
  $: maxRiskCount = Math.max(1, riskCounts.high, riskCounts.uncertain, riskCounts.low);
  $: topCases = [...cases].sort((a, b) => b.risk_score - a.risk_score).slice(0, 5);

  function pct(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  load();
</script>

<section class="band">
  <h1 class="section-title">Case Queue</h1>
  <p class="muted">
    Cases exist only for uncertain or high-risk scores. This queue is the human review surface, not
    the fraud detector itself.
  </p>
  <div class="metric-row">
    <div class="metric"><strong>{cases.length}</strong><span>total cases</span></div>
    <div class="metric"><strong>{pending}</strong><span>pending review</span></div>
    <div class="metric"><strong>{reviewed}</strong><span>reviewed</span></div>
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
    <div class="dashboard-grid">
      <section class="panel">
        <h2 class="section-title">Risk Mix</h2>
        {#each Object.entries(riskCounts) as [band, count]}
          <div class="bar-row">
            <span><span class={`pill ${band}`}>{band}</span></span>
            <div class="bar-track">
              <span style={`width: ${(count / maxRiskCount) * 100}%`}></span>
            </div>
            <strong>{count}</strong>
          </div>
        {/each}
      </section>
      <section class="panel">
        <h2 class="section-title">Highest Model Scores</h2>
        {#each topCases as item}
          <a class="case-card" href={`/cases/${item.case_id}`}>
            <span>{item.case_id.slice(0, 8)}</span>
            <strong>{pct(item.risk_score)}</strong>
            <span class={`pill ${item.status}`}>{item.status}</span>
          </a>
        {/each}
      </section>
    </div>

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
              risk {pct(item.risk_score)}<br />
              anomaly {pct(item.anomaly_score)}
            </td>
            <td>{item.created_at ? new Date(item.created_at).toLocaleString() : 'n/a'}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</section>
