<script lang="ts">
  import { listPredictions } from '$lib/api';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import ScoreBar from '$lib/components/ScoreBar.svelte';
  import Stat from '$lib/components/Stat.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';
  import type { PredictionHistoryItem, RiskBand } from '$lib/types';
  import { page } from '$app/stores';
  import { onMount } from 'svelte';

  type Filter = 'all' | 'audit-only' | 'review-cases' | RiskBand;

  let predictions: PredictionHistoryItem[] = [];
  let loading = true;
  let error = '';
  let filter: Filter = 'all';
  let sort = 'created_desc';

  async function load() {
    loading = true;
    error = '';
    try {
      predictions = await listPredictions({ limit: 300 });
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    const requested = $page.url.searchParams.get('filter');
    if (requested && FILTERS.some(([value]) => value === requested)) {
      filter = requested as Filter;
    }
    void load();
  });

  function pct(value: number) {
    return `${(value * 100).toFixed(1)}%`;
  }

  function fmtDate(value: string | null) {
    if (!value) return '-';
    return new Date(value).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  }

  function shortId(id: string) {
    return id.slice(0, 8);
  }

  function clamp(value: number) {
    return Math.max(0, Math.min(1, value));
  }

  function plotX(value: number) {
    return 28 + clamp(value) * 260;
  }

  function plotY(value: number) {
    return 148 - clamp(value) * 120;
  }

  function routeLabel(row: PredictionHistoryItem) {
    return row.case_id ? 'Review case' : 'Audit-only';
  }

  $: auditOnly = predictions.filter((row) => !row.case_id).length;
  $: reviewCases = predictions.filter((row) => row.case_id).length;
  $: highRisk = predictions.filter((row) => row.risk_band === 'high').length;
  $: uncertainRisk = predictions.filter((row) => row.risk_band === 'uncertain').length;
  $: riskMix = {
    low: predictions.filter((row) => row.risk_band === 'low').length,
    uncertain: uncertainRisk,
    high: highRisk
  };
  $: maxRisk = Math.max(1, ...Object.values(riskMix));
  $: filtered = predictions
    .filter((row) => {
      if (filter === 'all') return true;
      if (filter === 'audit-only') return !row.case_id;
      if (filter === 'review-cases') return !!row.case_id;
      return row.risk_band === filter;
    })
    .sort((a, b) => {
      if (sort === 'risk_desc') return b.risk_score - a.risk_score;
      if (sort === 'risk_asc') return a.risk_score - b.risk_score;
      if (sort === 'created_asc') {
        return new Date(a.created_at ?? 0).getTime() - new Date(b.created_at ?? 0).getTime();
      }
      return new Date(b.created_at ?? 0).getTime() - new Date(a.created_at ?? 0).getTime();
    });

  const FILTERS: [Filter, string][] = [
    ['all', 'All predictions'],
    ['audit-only', 'Audit-only'],
    ['review-cases', 'Review cases'],
    ['low', 'Low'],
    ['uncertain', 'Uncertain'],
    ['high', 'High']
  ];

  const RISK_COLOR: Record<RiskBand, string> = {
    high: 'var(--high)',
    uncertain: 'var(--uncertain)',
    low: 'var(--accent)'
  };
</script>

<div class="page">
  <div class="row-between" style="margin-bottom: 20px; align-items: flex-start;">
    <div>
      <h1 class="page-title">Prediction History</h1>
      <p class="page-sub">
        Every score lands here, including low-risk audit records that do not belong in Case Queue.
      </p>
    </div>
    <button class="btn" on:click={load}>Refresh</button>
  </div>

  {#if loading}
    <div style="text-align: center; padding: 48px;">
      <div class="spinner-lg"></div>
      <p class="muted" style="font-size: 13px;">Loading predictions...</p>
    </div>

  {:else if error}
    <div class="notice-error">{error}</div>

  {:else}
    <div class="grid-4" style="margin-bottom: 20px;">
      <Stat label="Total predictions" value={predictions.length} />
      <Stat label="Audit-only" value={auditOnly} accent="var(--accent)" />
      <Stat label="Review cases" value={reviewCases} accent={reviewCases > 0 ? 'var(--uncertain)' : 'var(--muted)'} />
      <Stat label="High risk" value={highRisk} accent={highRisk > 0 ? 'var(--high)' : 'var(--muted)'} />
    </div>

    <div class="grid-cases-dash">
      <div class="card">
        <SectionLabel>Risk and anomaly matrix</SectionLabel>
        {#if filtered.length === 0}
          <p class="muted" style="font-size: 12px;">No predictions match this filter.</p>
        {:else}
          <svg viewBox="0 0 320 180" role="img" aria-label="Risk and anomaly scatter plot" style="width: 100%; height: 220px; display: block;">
            <line x1="28" y1="148" x2="288" y2="148" stroke="var(--border)" />
            <line x1="28" y1="28" x2="28" y2="148" stroke="var(--border)" />
            <line x1="158" y1="28" x2="158" y2="148" stroke="rgba(255,255,255,.08)" stroke-dasharray="4 4" />
            <line x1="28" y1="88" x2="288" y2="88" stroke="rgba(255,255,255,.08)" stroke-dasharray="4 4" />
            <text x="28" y="168" fill="var(--muted)" font-size="9">0 risk</text>
            <text x="248" y="168" fill="var(--muted)" font-size="9">1.0 risk</text>
            <text x="2" y="32" fill="var(--muted)" font-size="9">anomaly</text>
            {#each filtered.slice(0, 180) as row}
              <circle
                cx={plotX(row.risk_score)}
                cy={plotY(row.anomaly_score)}
                r={row.case_id ? 4 : 3}
                fill={RISK_COLOR[row.risk_band]}
                opacity={row.case_id ? 0.9 : 0.55}
              />
            {/each}
          </svg>
          <p class="muted" style="font-size: 11px; line-height: 1.5;">
            Dots show model risk on the horizontal axis and autoencoder anomaly on the vertical axis.
            Larger dots opened review cases.
          </p>
        {/if}
      </div>

      <div class="card">
        <SectionLabel>Risk mix</SectionLabel>
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
        <div class="notice" style="margin-top: 14px;">
          Case Queue is intentionally review-focused. This page is the complete scoring ledger.
        </div>
      </div>
    </div>

    <div class="row" style="margin-bottom: 12px;">
      <span class="muted" style="font-size: 11px; margin-right: 4px;">Filter:</span>
      {#each FILTERS as [value, label]}
        <button class="filter-pill" class:active={filter === value} on:click={() => (filter = value)}>
          {label}
        </button>
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

    <div class="card" style="padding: 0; overflow: hidden;">
      {#if filtered.length === 0}
        <p class="muted" style="padding: 32px; text-align: center; font-size: 13px;">
          No predictions match this filter. Score a transaction or upload the demo CSV to populate history.
        </p>
      {:else}
        <table class="data-table">
          <thead>
            <tr>
              <th>Prediction</th>
              <th>Band</th>
              <th>Scores</th>
              <th>Amount</th>
              <th>Time</th>
              <th>Route</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>
            {#each filtered as row}
              <tr>
                <td><span class="mono muted" style="font-size: 12px;">{shortId(row.prediction_id)}</span></td>
                <td><RiskPill band={row.risk_band} /></td>
                <td style="min-width: 180px;">
                  <div class="stack" style="gap: 4px;">
                    <ScoreBar value={row.risk_score} band={row.risk_band} label="risk" />
                    <ScoreBar value={row.anomaly_score} band={row.risk_band} label="anomaly" />
                  </div>
                </td>
                <td><span class="mono" style="font-size: 12px;">{row.amount === null ? '-' : row.amount.toFixed(2)}</span></td>
                <td><span class="mono muted" style="font-size: 12px;">{row.transaction_time === null ? '-' : row.transaction_time.toFixed(0)}</span></td>
                <td>
                  {#if row.case_id}
                    <a href="/cases/{row.case_id}" class="link-accent" style="font-size: 12px;">{routeLabel(row)}</a>
                    {#if row.case_status}
                      <div style="margin-top: 4px;"><RiskPill status={row.case_status} /></div>
                    {/if}
                  {:else}
                    <span class="muted" style="font-size: 12px;">{routeLabel(row)}</span>
                  {/if}
                </td>
                <td><span class="muted" style="font-size: 12px; white-space: nowrap;">{fmtDate(row.created_at)}</span></td>
              </tr>
            {/each}
          </tbody>
        </table>
      {/if}
    </div>
  {/if}
</div>
