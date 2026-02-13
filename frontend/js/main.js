document.addEventListener('DOMContentLoaded', () => {
    console.log('Initializing application...');

    initThreeJS();

    document.getElementById('visualize-btn').addEventListener('click', fetchVisualization);

    const leftPanel = document.getElementById('controls');
    const rightPanel = document.getElementById('text-panel');
    const container = document.getElementById('container');
    const body = document.body;

    let leftVisible = true;
    let rightVisible = true;

    document.getElementById('toggle-left').addEventListener('click', () => {
        leftVisible = !leftVisible;

        if (leftVisible) {
            leftPanel.classList.remove('panel-hidden');
            body.classList.remove('left-panel-hidden');
            container.classList.remove('fullscreen-left', 'fullscreen-both');
            if (!rightVisible) {
                container.classList.add('fullscreen-right');
            }
        } else {
            leftPanel.classList.add('panel-hidden');
            body.classList.add('left-panel-hidden');
            container.classList.add('fullscreen-left');
            if (!rightVisible) {
                container.classList.remove('fullscreen-left');
                container.classList.add('fullscreen-both');
            } else {
                container.classList.remove('fullscreen-right');
            }
        }

        setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
    });

    document.getElementById('toggle-right').addEventListener('click', () => {
    // Check if there's actual content to show
    const selectedText = document.getElementById('selected-text');
    const defaultText = "Hover over a point to see details. Click to read the full clinical note.";
    const hasContent = selectedText.innerHTML.trim() !== defaultText.trim();

    // Only toggle if a point has been clicked
    if (!hasContent) {
        // Show helpful message
        selectedText.innerHTML = "⚠️ <strong>Click on a data point</strong> in the visualization to view its clinical note.";
        selectedText.style.color = "#FEC5F6";

        // Reset after 3 seconds
        setTimeout(() => {
            selectedText.innerHTML = defaultText;
            selectedText.style.color = "";
        }, 3000);

        return; // Don't toggle the panel
    }

    // If there's content, proceed with toggle
    rightVisible = !rightVisible;

    if (rightVisible) {
        rightPanel.classList.remove('panel-hidden');
        body.classList.remove('right-panel-hidden');
        container.classList.remove('fullscreen-right', 'fullscreen-both');
        if (!leftVisible) {
            container.classList.add('fullscreen-left');
        }
    } else {
        rightPanel.classList.add('panel-hidden');
        body.classList.add('right-panel-hidden');
        container.classList.add('fullscreen-right');
        if (!leftVisible) {
            container.classList.remove('fullscreen-right');
            container.classList.add('fullscreen-both');
        } else {
            container.classList.remove('fullscreen-left');
        }
    }

    setTimeout(() => window.dispatchEvent(new Event('resize')), 300);
});
    fetchVisualization();

    console.log('✓ Application ready');
});