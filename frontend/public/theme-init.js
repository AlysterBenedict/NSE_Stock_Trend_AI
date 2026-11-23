(function () {
  try {
    if (window.localStorage.getItem('neurostock-theme') === 'dark') {
      document.documentElement.setAttribute('data-theme', 'dark');
      document.documentElement.style.colorScheme = 'dark';
      var meta = document.querySelector('meta[name="theme-color"]');
      if (meta) meta.setAttribute('content', '#0b0f17');
    }
  } catch (e) {
    // Ignore storage access errors in restricted browser contexts
  }
})();
