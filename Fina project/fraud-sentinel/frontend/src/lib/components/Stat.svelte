<!--
  Stat.svelte
  Usage:
    <Stat label="Total cases" value={12} />
    <Stat label="Escalated" value={3} accent="var(--high)" />
-->
<script lang="ts">
  export let label: string;
  export let value: string | number;
  export let accent: string = 'var(--text)';
</script>

<div class="stat-card">
  <div class="stat-value" style="color: {accent}">{value}</div>
  <div class="stat-label">{label}</div>
</div>
