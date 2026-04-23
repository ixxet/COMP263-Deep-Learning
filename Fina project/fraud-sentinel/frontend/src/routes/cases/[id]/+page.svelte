<script lang="ts">
  import { getCase, reviewCase } from '$lib/api';
  import type { CaseDetail } from '$lib/types';
  import { page } from '$app/stores';
  import RiskPill from '$lib/components/RiskPill.svelte';
  import ScoreBar from '$lib/components/ScoreBar.svelte';
  import Stat from '$lib/components/Stat.svelte';
  import SectionLabel from '$lib/components/SectionLabel.svelte';
  import { onMount } from 'svelte';

  // ── State ──────────────────────────────────────────────────────────
  let item: CaseDetail | null = null;
  let loading = true;
  let error = '';
  let reviewer = 'analyst';
  let rationale = '';
  let submitting = false;
  let submitted = false;
  let activeAction: 'approve' | 'escalate' | 'dismiss' | null = null;

  // ── Load ───────────────────────────────────────────────────────────
  async function load() {
    loading = true; error = '';
    try {
      const id = $page.params.id;
      if (!id) throw new Error('Case ID missing from URL.');
      item = await getCase(id);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  onMount(() => {
    void load();
  });

  // ── Submit review ──────────────────────────────────────────────────
  async function submit() {
    if (!item || !activeAction || !rationale.trim()) return;
    submitting = true; error = '';
    try {
      await reviewCase(item.case_id, { action: activeAction, reviewer, rationale });
      rationale = '';
      activeAction = null;
      submitted = true;
      await load(); // reload to get updated status, reviews, audit_events
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      submitting = false;
    }
  }

  // ── Helpers ────────────────────────────────────────────────────────
  function pct(v: number) { return `${(v * 100).toFixed(1)}%`; }

  function fmtDate(s: string | undefined | null) {
    if (!s) return '—';
    return new Date(s).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
  }

  function riskColor(band: string) {
    return band === 'high' ? 'var(--high)' : band === 'uncertain' ? 'var(--uncertain)' : 'var(--accent)';
  }

  function eventName(event: Record<string, unknown>) {
    return String(event.event ?? event.event_type ?? '').replace(/_/g, ' ');
  }

  function eventTime(event: Record<string, unknown>) {
    return String(event.ts ?? event.created_at ?? '');
  }

  function reviewTime(review: Record<string, unknown>) {
    return String(review.ts ?? review.created_at ?? '');
  }

  // ── Action config ──────────────────────────────────────────────────
  const ACTIONS = [
    {
      key:   'approve'  as const,
      label: 'Approve flag',
      color: 'var(--accent)',
      desc:  'Confirm the model flag. Transaction pattern is consistent with fraud. Case closed as reviewed.',
    },
    {
      key:   'escalate' as const,
      label: 'Escalate',
      color: 'var(--uncertain)',
      desc:  'Send to deeper investigation or fraud operations. Pattern warrants further review.',
    },
    {
      key:   'dismiss'  as const,
      label: 'Dismiss',
      color: 'var(--muted)',
      desc:  'Close as false alarm or insufficient evidence. Record reasoning for audit trail.',
    },
  ];

  // ── Derived ────────────────────────────────────────────────────────
  $: canReview = !!item && ['pending_review', 'awaiting_human_review'].includes(item.status);
  $: canSubmit  = !!activeAction && rationale.trim().length > 0;
  $: agentBriefStatus = item?.brief
    ? 'brief_ready'
    : canReview
      ? 'agent_pending'
      : 'reviewed_no_brief';

  $: primaryTxFields = item
    ? Object.entries(item.transaction).filter(([k]) => ['Time', 'Amount'].includes(k))
    : [];
  $: pcaTxFields = item
    ? Object.entries(item.transaction).filter(([k]) => k.startsWith('V'))
    : [];
</script>

{#if loading}
  <div style="padding: 28px; text-align: center;">
    <div class="spinner-lg"></div>
    <p class="muted" style="font-size: 13px;">Loading case...</p>
  </div>

{:else if error && !item}
  <div style="padding: 28px;"><div class="notice-error">{error}</div></div>

{:else if item}
  <div class="grid-sidebar">

    <!-- ── Left: main review panel ──────────────────────────────────── -->
    <div style="overflow-y: auto; padding: 28px; border-right: 1px solid var(--border);">

      <!-- Breadcrumb -->
      <div class="row" style="margin-bottom: 20px;">
        <a href="/cases" class="btn btn-ghost" style="padding: 6px 12px; font-size: 12px;">← Cases</a>
        <span class="muted">|</span>
        <span class="mono muted" style="font-size: 12px;">{item.case_id}</span>
        <RiskPill status={item.status} />
      </div>

      <h1 class="page-title">Analyst Review</h1>
      <p class="page-sub">
        The model opened this case. Your decision and rationale are the analyst record — they
        persist in the audit trail regardless of action taken.
      </p>

      <!-- Scores -->
      <div class="grid-3" style="margin-bottom: 20px;">
        <Stat
          label="Supervised fraud score"
          value={pct(item.risk_score)}
          accent={riskColor(item.risk_band)}
        />
        <Stat
          label="Autoencoder anomaly"
          value={pct(item.anomaly_score)}
          accent={riskColor(item.risk_band)}
        />
        <div class="stat-card">
          <div style="margin-bottom: 8px;"><RiskPill band={item.risk_band} /></div>
          <div class="stat-label">risk band</div>
        </div>
      </div>

      <!-- Agent brief -->
      <div class="card" style="margin-bottom: 16px;">
        <div class="row-between" style="margin-bottom: 12px; align-items: center;">
          <SectionLabel>Agent brief</SectionLabel>
          <RiskPill status={agentBriefStatus} />
        </div>
        {#if item.brief}
          <p style="font-size: 13px; line-height: 1.7; color: var(--text);">{item.brief}</p>
          {#if item.policy_context.length > 0}
            <div class="stack" style="gap: 6px; margin-top: 12px;">
              {#each item.policy_context as ctx, i}
                <div style="
                  display: flex; align-items: center; gap: 8px;
                  padding: 6px 10px; background: var(--surface);
                  border-radius: var(--radius-sm); font-size: 11px;
                ">
                  <span style="color: var(--accent);">⊡</span>
                  <span style="font-weight: 500;">{String(ctx.title ?? 'Policy')}</span>
                  <span class="mono muted" style="margin-left: auto;">{String(ctx.source ?? 'local policy corpus')}</span>
                </div>
              {/each}
            </div>
          {/if}
        {:else}
          <div style="
            padding: 14px 16px; background: var(--surface);
            border-radius: var(--radius-md); border-left: 3px solid var(--border);
            font-size: 12px; color: var(--muted);
          ">
            Agent worker has not produced a brief yet. Case can still be reviewed from model scores,
            transaction payload, and analyst rationale.
          </div>
        {/if}
      </div>

      <!-- Decision panel (only when reviewable) -->
      {#if canReview && !submitted}
        <div class="card" style="border-color: rgba(20,184,166,.33); margin-bottom: 16px;">
          <SectionLabel>Analyst decision</SectionLabel>
          <p class="muted" style="font-size: 12px; margin-bottom: 16px; line-height: 1.5;">
            Select an action, then enter your rationale. Both are required and will persist in the audit trail.
          </p>

          <!-- Action selector -->
          <div class="grid-3" style="margin-bottom: 16px;">
            {#each ACTIONS as action}
              <button
                class="action-btn"
                class:selected-approve={activeAction === 'approve' && action.key === 'approve'}
                class:selected-escalate={activeAction === 'escalate' && action.key === 'escalate'}
                class:selected-dismiss={activeAction === 'dismiss' && action.key === 'dismiss'}
                on:click={() => (activeAction = activeAction === action.key ? null : action.key)}
              >
                <div class="action-btn-name">{action.label}</div>
                <div class="action-btn-desc">{action.desc}</div>
              </button>
            {/each}
          </div>

          <!-- Reviewer + rationale -->
          <div class="grid-2" style="margin-bottom: 12px;">
            <label class="field-label">
              <span>Reviewer ID</span>
              <input type="text" bind:value={reviewer} />
            </label>
          </div>

          <label class="field-label" style="margin-bottom: 16px;">
            <span>Rationale <span style="color: var(--high);">*</span></span>
            <textarea
              bind:value={rationale}
              placeholder="Record your analyst reasoning for this action. This becomes part of the permanent audit record."
            ></textarea>
          </label>

          {#if activeAction}
            {@const cfg = ACTIONS.find(a => a.key === activeAction)}
            <button
              class="btn"
              style="
                background: {canSubmit ? cfg?.color : 'var(--border)'};
                border-color: {canSubmit ? cfg?.color : 'var(--border)'};
                color: {canSubmit ? '#0d0f14' : 'var(--muted)'};
                font-weight: 700;
                cursor: {canSubmit ? 'pointer' : 'not-allowed'};
              "
              disabled={!canSubmit || submitting}
              on:click={submit}
            >
              {#if submitting}<span class="spinner"></span>{/if}
              {submitting ? 'Recording...' : `Record: ${cfg?.label}`}
            </button>
          {/if}

          {#if error}
            <div class="notice-error" style="margin-top: 12px;">{error}</div>
          {/if}
        </div>
      {/if}

      <!-- Success notice -->
      {#if submitted}
        <div class="notice-success" style="margin-bottom: 16px;">
          <span style="font-size: 18px;">✓</span>
          Decision recorded. Audit trail updated.
        </div>
      {/if}

      <!-- Closed state -->
      {#if !canReview && !submitted}
        <div class="card" style="background: var(--surface); margin-bottom: 16px;">
          <SectionLabel>Case closed</SectionLabel>
          <p style="font-size: 13px; color: var(--muted);">
            This case has been reviewed. Status: <RiskPill status={item.status} />
          </p>
        </div>
      {/if}

      <!-- Review history -->
      {#if item.reviews.length > 0}
        <div style="margin-top: 4px;">
          <SectionLabel>Review history</SectionLabel>
          <div class="stack">
            {#each item.reviews as review}
              <div class="card" style="border-left: 3px solid var(--accent); padding: 12px 16px;">
                <div class="row" style="margin-bottom: 6px;">
                  <RiskPill status={String(review.action)} />
                  <span class="mono muted" style="font-size: 11px;">{String(review.reviewer)}</span>
                  <span class="muted" style="font-size: 11px; margin-left: auto;">{fmtDate(reviewTime(review))}</span>
                </div>
                <p style="font-size: 12px; line-height: 1.5;">{String(review.rationale)}</p>
              </div>
            {/each}
          </div>
        </div>
      {/if}
    </div>

    <!-- ── Right: context panel ──────────────────────────────────────── -->
    <div style="overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 14px;">

      <!-- Audit timeline -->
      <div class="card">
        <SectionLabel>Audit timeline</SectionLabel>
        <div class="timeline">
          <div class="timeline-line"></div>
          {#each item.audit_events as evt, i}
            <div class="timeline-item">
              <div
                class="timeline-dot"
                class:active={i === item.audit_events.length - 1}
                class:past={i !== item.audit_events.length - 1}
              ></div>
              <div class="timeline-event">{eventName(evt)}</div>
              <div class="timeline-ts">{fmtDate(eventTime(evt))}</div>
            </div>
          {/each}
        </div>
      </div>

      <!-- Transaction payload -->
      <div class="card">
        <SectionLabel>Transaction payload</SectionLabel>

        {#each primaryTxFields as [key, value]}
          <div style="display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px solid rgba(45,49,71,.4);">
            <span class="muted" style="font-size: 12px;">{key}</span>
            <span class="mono" style="font-size: 12px;">{Number(value).toFixed(4)}</span>
          </div>
        {/each}

        {#if pcaTxFields.length > 0}
          <details style="margin-top: 10px;">
            <summary class="muted" style="font-size: 11px; cursor: pointer; padding: 4px 0; list-style: none;">
              ▸ V1–V28 model features ({pcaTxFields.length} fields)
            </summary>
            <div style="margin-top: 8px; display: grid; grid-template-columns: 1fr 1fr; gap: 2px;">
              {#each pcaTxFields as [key, value]}
                {@const v = Number(value)}
                <div style="display: flex; justify-content: space-between; padding: 3px 0;">
                  <span class="mono muted" style="font-size: 10px;">{key}</span>
                  <span
                    class="mono"
                    style="font-size: 10px; color: {v < -2 ? 'var(--high)' : v > 2 ? 'var(--uncertain)' : 'var(--text)'};"
                  >{v.toFixed(3)}</span>
                </div>
              {/each}
            </div>
          </details>
        {/if}
      </div>

      <!-- Decision guide -->
      <div class="card" style="background: var(--surface);">
        <SectionLabel>Decision guide</SectionLabel>
        <div class="stack" style="gap: 6px; font-size: 11px; color: var(--muted);">
          {#each [
            { color: 'var(--accent)',    label: 'Approve',  desc: 'Model flag confirmed. Pattern consistent with known fraud.' },
            { color: 'var(--uncertain)', label: 'Escalate', desc: 'Needs deeper investigation. Refer to fraud operations.' },
            { color: 'var(--muted)',     label: 'Dismiss',  desc: 'False alarm or insufficient evidence. Record reasoning.' },
          ] as row}
            <div style="padding: 6px 10px; background: var(--card); border-radius: var(--radius-sm); border-left: 3px solid {row.color};">
              <strong style="color: {row.color};">{row.label}</strong> — {row.desc}
            </div>
          {/each}
        </div>
      </div>

    </div>
  </div>
{/if}
