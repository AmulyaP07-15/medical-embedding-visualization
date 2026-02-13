// Global variables
let scene, camera, renderer, controls;
let points = [];
let pointsObject;
let raycaster, mouse;
let hoveredPoint = null;

// Specialty colors
const SPECIALTY_COLORS = {
    'Surgery': 0xff0066,
    'Cardiovascular / Pulmonary': 0x00ffff,
    'Orthopedic': 0xffff00,
    'Radiology': 0x00ff88,
    'Neurology': 0xff6600,
    'Gastroenterology': 0xff00ff
};

function initThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x000000);
    scene.fog = new THREE.FogExp2(0x000000, 0.02);
    
    const container = document.getElementById('visualization');
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        1000
    );
    camera.position.set(0, 5, 15);
    
    renderer = new THREE.WebGLRenderer({ 
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);
    
    // Add grid helper (cyan glow)
    const gridHelper = new THREE.GridHelper(30, 30, 0x00d9ff, 0x003344);
    gridHelper.position.y = -5;
    scene.add(gridHelper);
    
    // Add axis helper in corner
    const axesHelper = new THREE.AxesHelper(3);
    axesHelper.position.set(-12, -4, -12);
    scene.add(axesHelper);
    
    // Try to use OrbitControls if available
    if (typeof THREE.OrbitControls !== 'undefined') {
        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        controls.minDistance = 5;
        controls.maxDistance = 40;
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.5;
        console.log('✓ OrbitControls enabled');
    } else {
        console.warn('⚠ OrbitControls not found, using manual controls');
        setupManualControls(container);
    }
    
    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);
    
    const light1 = new THREE.PointLight(0x00ffff, 1.5, 100);
    light1.position.set(15, 15, 15);
    scene.add(light1);
    
    const light2 = new THREE.PointLight(0xff00ff, 1.5, 100);
    light2.position.set(-15, -15, -15);
    scene.add(light2);
    
    // Raycaster for hover
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.3;
    mouse = new THREE.Vector2();
    
    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('click', onMouseClick);
    
    window.addEventListener('resize', onWindowResize);
    
    animate();
    
    console.log('✓ Three.js initialized with grid');
}

function setupManualControls(container) {
    // Manual zoom with scroll
    let cameraDistance = 15;
    
    container.addEventListener('wheel', (e) => {
        e.preventDefault();
        cameraDistance += e.deltaY * 0.01;
        cameraDistance = Math.max(5, Math.min(40, cameraDistance));
        
        const direction = camera.position.clone().normalize();
        camera.position.copy(direction.multiplyScalar(cameraDistance));
        camera.lookAt(0, 0, 0);
    }, { passive: false });
    
    // Manual rotation with drag
    let isDragging = false;
    let previousMouse = { x: 0, y: 0 };
    let rotation = { x: 0, y: 0 };
    
    container.addEventListener('mousedown', (e) => {
        if (e.button === 0) { // Left click
            isDragging = true;
            previousMouse = { x: e.clientX, y: e.clientY };
        }
    });
    
    container.addEventListener('mouseup', () => {
        isDragging = false;
    });
    
    container.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const deltaX = e.clientX - previousMouse.x;
            const deltaY = e.clientY - previousMouse.y;
            
            rotation.y += deltaX * 0.01;
            rotation.x += deltaY * 0.01;
            
            // Update camera position
            const distance = camera.position.length();
            camera.position.x = distance * Math.sin(rotation.y) * Math.cos(rotation.x);
            camera.position.y = distance * Math.sin(rotation.x);
            camera.position.z = distance * Math.cos(rotation.y) * Math.cos(rotation.x);
            camera.lookAt(0, 0, 0);
            
            previousMouse = { x: e.clientX, y: e.clientY };
        }
    });
    
    console.log('✓ Manual controls enabled');
}

function onWindowResize() {
    const container = document.getElementById('visualization');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function onMouseMove(event) {
    const container = document.getElementById('visualization');
    const rect = container.getBoundingClientRect();
    
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
}

function onMouseClick(event) {
    if (hoveredPoint) {
        document.getElementById('selected-text').innerHTML = `
            <strong>Specialty:</strong> ${hoveredPoint.specialty}<br><br>
            ${hoveredPoint.full_text}
        `;
    }
}

function animate() {
    requestAnimationFrame(animate);
    
    if (controls) {
        controls.update();
    }
    
    // Check for hover
    if (pointsObject && points.length > 0) {
        raycaster.setFromCamera(mouse, camera);
        
        // Check all children if pointsObject is a group
        let intersects = [];
        if (pointsObject.children && pointsObject.children.length > 0) {
            intersects = raycaster.intersectObjects(pointsObject.children);
        } else {
            intersects = raycaster.intersectObject(pointsObject);
        }
        
        if (intersects.length > 0) {
            const index = pointsObject.children ? 
                pointsObject.children.indexOf(intersects[0].object) : 
                intersects[0].index;
                
            if (index >= 0 && index < points.length) {
                hoveredPoint = points[index];
                showTooltip(hoveredPoint);
            }
        } else {
            hoveredPoint = null;
            hideTooltip();
        }
    }
    
    renderer.render(scene, camera);
}

function showTooltip(point) {
    const tooltip = document.getElementById('tooltip');
    const container = document.getElementById('visualization');
    const rect = container.getBoundingClientRect();
    
    tooltip.style.display = 'block';
    tooltip.style.left = (mouse.x * 0.5 + 0.5) * rect.width + rect.left + 'px';
    tooltip.style.top = (-mouse.y * 0.5 + 0.5) * rect.height + rect.top + 'px';
    tooltip.innerHTML = `
        <strong style="color: #${SPECIALTY_COLORS[point.specialty].toString(16)}">${point.specialty}</strong><br>
        <span style="font-size: 11px; color: #aaa;">${point.text_snippet}</span>
    `;
}

function hideTooltip() {
    document.getElementById('tooltip').style.display = 'none';
}

function visualizePoints(data) {
    console.log(`Visualizing ${data.points.length} points`);
    
    if (pointsObject) {
        scene.remove(pointsObject);
    }
    
    const coords = data.points.map(p => [p.x, p.y, p.z]);
    const { normalized } = normalizeCoordinates(coords, 10);
    
    points = data.points.map((point, i) => ({
        ...point,
        x: normalized[i][0],
        y: normalized[i][1],
        z: normalized[i][2]
    }));
    
    // Create smooth sphere particles
    const sphereGeometry = new THREE.SphereGeometry(0.15, 16, 16);
    pointsObject = new THREE.Group();
    
    points.forEach(point => {
        const color = new THREE.Color(SPECIALTY_COLORS[point.specialty] || 0xffffff);
        
        const material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.5,
            metalness: 0.5,
            roughness: 0.5
        });
        
        const sphere = new THREE.Mesh(sphereGeometry, material);
        sphere.position.set(point.x, point.y, point.z);
        
        pointsObject.add(sphere);
    });
    
    scene.add(pointsObject);
    
    updateLegend();
    document.getElementById('loading').style.display = 'none';
    
    console.log('✓ Smooth visualization complete');
}

function normalizeCoordinates(coords, spreadFactor = 10) {
    const mins = [Infinity, Infinity, Infinity];
    const maxs = [-Infinity, -Infinity, -Infinity];
    
    coords.forEach(([x, y, z]) => {
        mins[0] = Math.min(mins[0], x);
        mins[1] = Math.min(mins[1], y);
        mins[2] = Math.min(mins[2], z);
        maxs[0] = Math.max(maxs[0], x);
        maxs[1] = Math.max(maxs[1], y);
        maxs[2] = Math.max(maxs[2], z);
    });
    
    const ranges = maxs.map((max, i) => max - mins[i]);
    const scale = Math.max(...ranges);
    
    const normalized = coords.map(([x, y, z]) => [
        ((x - mins[0]) / scale) * spreadFactor - spreadFactor/2,
        ((y - mins[1]) / scale) * spreadFactor - spreadFactor/2,
        ((z - mins[2]) / scale) * spreadFactor - spreadFactor/2
    ]);
    
    return { normalized, scale };
}

function updateLegend() {
    const legendItems = document.getElementById('legend-items');
    legendItems.innerHTML = '';
    
    Object.entries(SPECIALTY_COLORS).forEach(([specialty, color]) => {
        const item = document.createElement('div');
        item.className = 'legend-item';
        
        const colorBox = document.createElement('div');
        colorBox.className = 'legend-color';
        const hexColor = '#' + color.toString(16).padStart(6, '0');
        colorBox.style.backgroundColor = hexColor;
        colorBox.style.boxShadow = `0 0 15px ${hexColor}`;
        
        const label = document.createElement('span');
        label.textContent = specialty;
        
        item.appendChild(colorBox);
        item.appendChild(label);
        legendItems.appendChild(item);
    });
}
