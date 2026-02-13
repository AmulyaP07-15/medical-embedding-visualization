const API_BASE_URL = 'http://127.0.0.1:8000/api';

async function fetchVisualization() {
    const modelId = document.getElementById('model-select').value;
    const reduction = document.getElementById('reduction-select').value;
    
    console.log(`Fetching ${modelId} with ${reduction}...`);
    
    // Show loading
    document.getElementById('loading').style.display = 'block';
    document.getElementById('loading').textContent = 
        reduction === 'umap' ? 'Computing UMAP (this may take 30-60 seconds)...' : 'Loading...';
    
    try {
        const response = await fetch(
            `${API_BASE_URL}/visualize/${modelId}?reduction=${reduction}`
        );
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('✓ Data fetched:', data.total_points, 'points');
        
        // Update stats
        document.getElementById('total-points').textContent = data.total_points;
        
        // Visualize
        visualizePoints(data);
        
    } catch (error) {
        console.error('Error fetching visualization:', error);
        document.getElementById('loading').textContent = 'Error loading data. Check console.';
    }
}
