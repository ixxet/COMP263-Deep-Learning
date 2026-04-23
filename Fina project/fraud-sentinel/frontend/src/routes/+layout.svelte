<script lang="ts">
  import '../app.css';
  import { page } from '$app/stores';
  import { browser } from '$app/environment';
  import { onMount } from 'svelte';

  $: currentPath = $page.url.pathname;
  let modelVersion = '';

  const modelVersionStorageKey = 'fraud-sentinel-model-version';
  const activePathStorageKey = 'fraud-sentinel-active-path';

  const navItems = [
    { href: '/',       icon: '⬡', label: 'Score Transaction' },
    { href: '/batch',  icon: '⊞', label: 'Batch Upload'      },
    { href: '/history', icon: '⋯', label: 'Prediction History' },
    { href: '/cases',  icon: '⊡', label: 'Case Queue'        },
  ];

  function isActive(href: string): boolean {
    if (href === '/') return currentPath === '/';
    return currentPath.startsWith(href);
  }

  function readModelVersion() {
    modelVersion = localStorage.getItem(modelVersionStorageKey) ?? '';
  }

  onMount(() => {
    readModelVersion();
    localStorage.setItem(activePathStorageKey, currentPath);

    const handleModelVersion = () => readModelVersion();
    window.addEventListener('fraud-sentinel-model-version', handleModelVersion);
    window.addEventListener('storage', handleModelVersion);

    return () => {
      window.removeEventListener('fraud-sentinel-model-version', handleModelVersion);
      window.removeEventListener('storage', handleModelVersion);
    };
  });

  $: if (browser) {
    localStorage.setItem(activePathStorageKey, currentPath);
  }
</script>

<div class="shell">
  <!-- Sidebar -->
  <aside class="sidebar">
    <div class="sidebar-header">
      <div class="brand">
        <div class="brand-mark">F</div>
        <div>
          <div class="brand-name">Fraud Sentinel</div>
          <div class="brand-sub">ANALYST CONSOLE</div>
        </div>
      </div>
    </div>

    <nav class="sidebar-nav" aria-label="Primary navigation">
      <div class="nav-section-label">Workbench</div>
      {#each navItems as item}
        <a
          href={item.href}
          class="nav-item"
          class:active={isActive(item.href)}
          aria-current={isActive(item.href) ? 'page' : undefined}
          on:click={() => localStorage.setItem(activePathStorageKey, item.href)}
        >
          <span class="nav-icon" aria-hidden="true">{item.icon}</span>
          {item.label}
        </a>
      {/each}
    </nav>

    <div class="sidebar-footer">
      <div class="status-dot" aria-hidden="true"></div>
      <span class="status-label">
        {modelVersion ? `Model ${modelVersion} · live` : 'Model awaiting score · live'}
      </span>
    </div>
  </aside>

  <!-- Main -->
  <main class="main">
    <slot />
  </main>
</div>
