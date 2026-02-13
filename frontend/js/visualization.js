// Global variables
let scene, camera, renderer;
let points = [];
let pointsObject;
let raycaster, mouse;
let hoveredPoint = null;
let isDragging = false;
let previousMousePosition = { x: 0, y: 0 };

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
    camera.lookAt(0, 0, 0);

    renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Add grid on XZ plane (horizontal floor)
    const gridHelper = new THREE.GridHelper(30, 30, 0x00d9ff, 0x003344);
    gridHelper.position.y = 0;
    scene.add(gridHelper);

    // Add full-size axes through the space
    const axisLength = 15;

    // X-axis (Red) - Left/Right
    const xAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-axisLength, 0, 0),
        new THREE.Vector3(axisLength, 0, 0)
    ]);
    const xAxisMaterial = new THREE.LineBasicMaterial({ color: 0xff0000, linewidth: 2 });
    const xAxis = new THREE.Line(xAxisGeometry, xAxisMaterial);
    scene.add(xAxis);

    // Y-axis (Green) - Up/Down
    const yAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, -axisLength, 0),
        new THREE.Vector3(0, axisLength, 0)
    ]);
    const yAxisMaterial = new THREE.LineBasicMaterial({ color: 0x00ff00, linewidth: 2 });
    const yAxis = new THREE.Line(yAxisGeometry, yAxisMaterial);
    scene.add(yAxis);

    // Z-axis (Blue) - Forward/Back
    const zAxisGeometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(0, 0, -axisLength),
        new THREE.Vector3(0, 0, axisLength)
    ]);
    const zAxisMaterial = new THREE.LineBasicMaterial({ color: 0x0000ff, linewidth: 2 });
    const zAxis = new THREE.Line(zAxisGeometry, zAxisMaterial);
    scene.add(zAxis);

    // Add grids on other planes for full 3D grid effect
    const gridYZ = new THREE.GridHelper(30, 30, 0x00d9ff, 0x002244);
    gridYZ.rotation.z = Math.PI / 2;
    gridYZ.position.x = 0;
    scene.add(gridYZ);

    const gridXY = new THREE.GridHelper(30, 30, 0x00d9ff, 0x002244);
    gridXY.rotation.x = Math.PI / 2;
    gridXY.position.z = 0;
    scene.add(gridXY);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
    scene.add(ambientLight);

    const light1 = new THREE.PointLight(0x00ffff, 1.5, 100);
    light1.position.set(15, 15, 15);
    scene.add(light1);

    const light2 = new THREE.PointLight(0xff00ff, 1.5, 100);
    light2.position.set(-15, -15, -15);
    scene.add(light2);

    // SIMPLE ZOOM - Mouse wheel
    container.addEventListener('wheel', function(event) {
        event.preventDefault();

        const zoomSpeed = 0.5;
        const direction = camera.position.clone().normalize();

        if (event.deltaY < 0) {
            // Zoom in
            camera.position.sub(direction.multiplyScalar(zoomSpeed));
        } else {
            // Zoom out
            camera.position.add(direction.multiplyScalar(zoomSpeed));
        }

        // Limit zoom distance
        const distance = camera.position.length();
        if (distance < 5) {
            camera.position.normalize().multiplyScalar(5);
        } else if (distance > 40) {
            camera.position.normalize().multiplyScalar(40);
        }

        camera.lookAt(0, 0, 0);

        console.log('Zoom - distance:', distance.toFixed(2));
    }, { passive: false });

    // DRAG TO ROTATE
    container.addEventListener('mousedown', function(e) {
        isDragging = true;
        previousMousePosition = { x: e.clientX, y: e.clientY };
    });

    container.addEventListener('mouseup', function() {
        isDragging = false;
    });

    container.addEventListener('mousemove', function(e) {
        if (isDragging) {
            const deltaX = e.clientX - previousMousePosition.x;
            const deltaY = e.clientY - previousMousePosition.y;

            const rotationSpeed = 0.005;

            // Rotate camera around Y axis (horizontal)
            const angleY = deltaX * rotationSpeed;
            const x = camera.position.x;
            const z = camera.position.z;
            camera.position.x = x * Math.cos(angleY) - z * Math.sin(angleY);
            camera.position.z = x * Math.sin(angleY) + z * Math.cos(angleY);

            // Rotate camera around X axis (vertical)
            const angleX = -deltaY * rotationSpeed;
            const y = camera.position.y;
            const distance = Math.sqrt(camera.position.x * camera.position.x + camera.position.z * camera.position.z);
            camera.position.y = y + distance * Math.sin(angleX);

            // Clamp vertical rotation
            camera.position.y = Math.max(-20, Math.min(20, camera.position.y));

            camera.lookAt(0, 0, 0);
            previousMousePosition = { x: e.clientX, y: e.clientY };
        }
    });

    // Raycaster for hover
    raycaster = new THREE.Raycaster();
    raycaster.params.Points.threshold = 0.3;
    mouse = new THREE.Vector2();

    container.addEventListener('mousemove', onMouseMove);
    container.addEventListener('click', onMouseClick);

    window.addEventListener('resize', onWindowResize);

    animate();

    console.log('✓ Three.js initialized with 3D grid and axes');
}

function onWindowResize() {
    const container = document.getElementById('visualization');
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function onMouseMove(event) {
    if (isDragging) return;

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

    // Gentle auto-rotation
    if (pointsObject && !isDragging) {
        const rotationSpeed = 0.001;
        const x = camera.position.x;
        const z = camera.position.z;
        camera.position.x = x * Math.cos(rotationSpeed) - z * Math.sin(rotationSpeed);
        camera.position.z = x * Math.sin(rotationSpeed) + z * Math.cos(rotationSpeed);
        camera.lookAt(0, 0, 0);
    }

    // Check for hover
    if (pointsObject && points.length > 0 && !isDragging) {
        raycaster.setFromCamera(mouse, camera);

        let intersects = [];
        if (pointsObject.children && pointsObject.children.length > 0) {
            intersects = raycaster.intersectObjects(pointsObject.children);
        }

        if (intersects.length > 0) {
            const index = pointsObject.children.indexOf(intersects[0].object);

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

    // Create smooth sphere particles - BIGGER and GLOWING
    const sphereGeometry = new THREE.SphereGeometry(0.25, 32, 32);
    pointsObject = new THREE.Group();

    points.forEach(point => {
        const color = new THREE.Color(SPECIALTY_COLORS[point.specialty] || 0xffffff);

        const material = new THREE.MeshStandardMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.8,
            metalness: 0.3,
            roughness: 0.2,
            transparent: true,
            opacity: 0.9
        });

        const sphere = new THREE.Mesh(sphereGeometry, material);
        sphere.position.set(point.x, point.y, point.z);

        pointsObject.add(sphere);
    });

    scene.add(pointsObject);

    updateLegend();
    document.getElementById('loading').style.display = 'none';

    console.log('✓ Smooth sphere visualization complete');
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