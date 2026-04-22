<script lang="ts">
  import '../app.css';
</script>

<div class="shell">
  <header class="topbar">
    <div class="brand">
      <div class="brand-mark">FS</div>
      <div>
        <div>Fraud Sentinel</div>
        <div class="muted">Model scoring and analyst review</div>
      </div>
    </div>
    <nav class="nav" aria-label="Primary">
      <a href="/">Predictions</a>
      <a href="/cases">Cases</a>
    </nav>
  </header>
  <main class="content">
    <slot />
  </main>
</div>

