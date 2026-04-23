<!--
  RiskPill.svelte
  Usage:
    <RiskPill band="high" />
    <RiskPill status="pending_review" />
  Pass either `band` or `status` — whichever is relevant.
-->
<script lang="ts">
  export let band: string = '';
  export let status: string = '';

  $: value = status || band;
  $: display = value.replace(/_/g, ' ');
</script>

<span class="pill {value}">
  <span class="pill-dot" style="background: currentColor"></span>
  {display}
</span>
