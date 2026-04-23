<!--
  ScoreBar.svelte
  Usage:
    <ScoreBar value={0.93} band="high" label="Supervised fraud score" />
  `value` is 0–1. `label` is optional.
-->
<script lang="ts">
  export let value: number;
  export let band: string = 'low';
  export let label: string = '';

  $: color =
    band === 'high'      ? 'var(--high)'      :
    band === 'uncertain' ? 'var(--uncertain)' :
                           'var(--accent)';

  $: pct = `${(value * 100).toFixed(1)}%`;
</script>

<div>
  {#if label}
    <div class="score-bar-label">{label}</div>
  {/if}
  <div class="score-bar-row">
    <div class="score-bar-track">
      <div class="score-bar-fill" style="width: {pct}; background: {color}"></div>
    </div>
    <span class="score-bar-value" style="color: {color}">{pct}</span>
  </div>
</div>
