<script lang="ts">
  import { listCases } from '$lib/api';
  import type { CaseSummary } from '$lib/types';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import ScoreBar from '$lib/components/ScoreBar.svelte';
  import Stat from '$lib/components/Stat.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';
  import { onMount } from 'svelte';

  // ── State ──────────────────────────────────────────────────────────
  let cases: CaseSummary[] = [];
  let loading = true;
  let error = '';
  let filter = 'all';
  let sort = 'created_desc';

  // ── Load ───────────────────────────────────────────────────────────
  async function load() {
    loading = true; error = '';
    try { cases = await listCases(); }
    catch (err) { error = err instanceof Error ? err.message : String(err); }
    finally { loading = false; }
  }

  onMount(() => {
    void load();
  });

  // ── Derived ────────────────────────────────────────────────────────
  $: pending   = cases.filter(c => ['pending_review','awaiting_human_review'].includes(c.status));
  $: reviewed  = cases.filter(c => ['approved','escalated','dismissed'].includes(c.status));
  $: escalated = cases.filter(c => c.status === 'escalated');

  $: riskMix = {
    high:      cases.filter(c => c.risk_band === 'high').length,
    uncertain: cases.filter(c => c.risk_band === 'uncertain').length,
    low:       cases.filter(c => c.risk_band === 'low').length,
  };
  $: maxRisk = Math.max(1, ...Object.values(riskMix));
  $: topCases = [...cases].sort((a, b) => b.risk_score - a.risk_score).slice(0, 4);

  $: filtered = cases
    .filter(c => {
      if (filter === 'all') return true;
      return c.status === filter || c.risk_band === filter;
    })
    .sort((a, b) => {
      if (sort === 'risk_desc') return b.risk_score - a.risk_score;
      if (sort === 'risk_asc')  return a.risk_score - b.risk_score;
      if (sort === 'created_asc') return new Date(a.created_at ?? 0).getTime() - new Date(b.created_at ?? 0).getTime();
      return new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime();
    });

  // ── Helpers ────────────────────────────────────────────────────────
  function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }
  function fmtDate(s: string | null) {
    if (!s) return '—';
    return new Date(s).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }
  function shortId(id: string) { return id.slice(0, 8); }

  const FILTERS = [
    ['all',                   'All cases'],
    ['pending_review',        'Pending'],
    ['awaiting_human_review', 'Awaiting review'],
    ['escalated',             'Escalated'],
    ['high',                  'High risk'],
    ['uncertain',             'Uncertain'],
  ] as const;

  const RISK_COLOR: Record<string, string> = {
    high:      'var(--high)',
    uncertain: 'var(--uncertain)',
    low:       'var(--accent)',
  };
</script>

<div class="page">
  <!-- Header -->
  <div class="row-between" style="margin-bottom: 20px; align-items: flex-start;">
    <div>
      <h1 class="page-title">Case Queue</h1>
      <p class="muted" style="font-size: 13px;">
        Cases exist only for uncertain or high-risk scores. This is the human review surface.
      </p>
    </div>
    <button class="btn" on:click={load}>↻ Refresh</button>
  </div>

  {#if loading}
    <div style="text-align: center; padding: 48px;">
      <div class="spinner-lg"></div>
      <p class="muted" style="font-size: 13px;">Loading cases...</p>
    </div>

  {:else if error}
    <div class="notice-error">{error}</div>

  {:else}
    <!-- Stats -->
    <div class="grid-4" style="margin-bottom: 20px;">
      <Stat label="Total cases"     value={cases.length} />
      <Stat label="Needs attention" value={pending.length}
        accent={pending.length > 0 ? 'var(--uncertain)' : 'var(--muted)'} />
      <Stat label="Reviewed"        value={reviewed.length}  accent="var(--accent)" />
      <Stat label="Escalated"       value={escalated.length} accent={escalated.length > 0 ? 'var(--high)' : 'var(--muted)'} />
    </div>

    <!-- Dashboard row -->
    <div class="grid-cases-dash">
      <!-- Risk mix -->
      <div class="card">
        <SectionLabel>Risk mix</SectionLabel>
        {#each Object.entries(riskMix) as [band, count]}
          <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
            <div style="width: 90px; flex-shrink: 0;">
              <RiskPill {band} />
            </div>
            <div style="flex: 1; height: 8px; background: rgba(255,255,255,.05); border-radius: 4px; overflow: hidden;">
              <div style="
                width: {(count / maxRisk) * 100}%;
                height: 100%;
                background: {RISK_COLOR[band]};
                border-radius: 4px;
                transition: width .6s ease;
              "></div>
            </div>
            <span class="mono" style="font-size: 13px; font-weight: 700; min-width: 24px; text-align: right;">{count}</span>
          </div>
        {/each}
      </div>

      <!-- Top risk scores -->
      <div class="card">
        <SectionLabel>Top risk scores</SectionLabel>
        {#each topCases as c}
          <a
            href="/cases/{c.case_id}"
            style="display: flex; align-items: center; gap: 10px; padding: 8px 0; border-bottom: 1px solid rgba(45,49,71,.5); text-decoration: none;"
          >
            <span class="mono muted" style="font-size: 11px; flex-shrink: 0;">{shortId(c.case_id)}</span>
            <div style="flex: 1;">
              <ScoreBar value={c.risk_score} band={c.risk_band} />
            </div>
            <RiskPill status={c.status} />
          </a>
        {/each}
      </div>
    </div>

    <!-- Filter + sort bar -->
    <div class="row" style="margin-bottom: 12px;">
      <span class="muted" style="font-size: 11px; margin-right: 4px;">Filter:</span>
      {#each FILTERS as [val, label]}
        <button
          class="filter-pill"
          class:active={filter === val}
          on:click={() => (filter = val)}
        >{label}</button>
      {/each}
      <div style="margin-left: auto; display: flex; align-items: center; gap: 6px;">
        <span class="muted" style="font-size: 11px;">Sort:</span>
        <select bind:value={sort} style="background: var(--card); border: 1px solid var(--border); color: var(--text); border-radius: 7px; padding: 4px 8px; font-size: 11px; cursor: pointer; outline: none;">
          <option value="created_desc">Newest first</option>
          <option value="created_asc">Oldest first</option>
          <option value="risk_desc">Highest risk</option>
          <option value="risk_asc">Lowest risk</option>
        </select>
      </div>
    </div>

    <!-- Cases table -->
    <div class="card" style="padding: 0; overflow: hidden;">
      {#if filtered.length === 0}
        <p class="muted" style="padding: 32px; text-align: center; font-size: 13px;">No cases match this filter.</p>
      {:else}
        <table class="data-table">
          <thead>
            <tr>
              <th>Case ID</th>
              <th>Status</th>
              <th>Risk band</th>
              <th>Model scores</th>
              <th>Created</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {#each filtered as c}
              <tr on:click={() => window.location.href = `/cases/${c.case_id}`}>
                <td><span class="mono muted" style="font-size: 12px;">{shortId(c.case_id)}</span></td>
                <td><RiskPill status={c.status} /></td>
                <td><RiskPill band={c.risk_band} /></td>
                <td style="min-width: 160px;">
                  <div class="stack" style="gap: 4px;">
                    <ScoreBar value={c.risk_score}    band={c.risk_band} label="risk" />
                    <ScoreBar value={c.anomaly_score} band={c.risk_band} label="anomaly" />
                  </div>
                </td>
                <td><span class="muted" style="font-size: 12px; white-space: nowrap;">{fmtDate(c.created_at)}</span></td>
                <td><span class="link-accent" style="font-size: 12px;">Review →</span></td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  {/if}
</div>
