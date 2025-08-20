// Set dark mode as default for Furo theme
document.addEventListener('DOMContentLoaded', function() {
    // Check if user has a saved preference
    const savedTheme = localStorage.getItem('theme');
    
    // If no saved preference, set to dark mode
    if (!savedTheme) {
        localStorage.setItem('theme', 'dark');
        document.documentElement.dataset.theme = 'dark';
        document.body.setAttribute('data-theme', 'dark');
    } else if (savedTheme === 'auto') {
        // If set to auto, override to dark
        localStorage.setItem('theme', 'dark');
        document.documentElement.dataset.theme = 'dark';
        document.body.setAttribute('data-theme', 'dark');
    }
});

// Also set it immediately (before DOMContentLoaded)
(function() {
    const savedTheme = localStorage.getItem('theme');
    if (!savedTheme || savedTheme === 'auto') {
        localStorage.setItem('theme', 'dark');
        document.documentElement.dataset.theme = 'dark';
    }
})();