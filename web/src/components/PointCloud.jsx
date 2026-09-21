import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function PointCloud({ cloud, nodes, mode, showBoxes, selectedId, onSelect, placementMode = false, placementPoint = null, placementZ = 0, onPlace, showPoints = true, showLabels = false }) {
  const mountRef = useRef(null)

  useEffect(() => {
    if (!cloud || !mountRef.current) return undefined
    const mount = mountRef.current
    const scene = new THREE.Scene()
    scene.background = new THREE.Color('#f8f8fa')
    scene.fog = new THREE.FogExp2('#f8f8fa', 0.035)

    const camera = new THREE.PerspectiveCamera(46, 1, 0.01, 100)
    camera.position.set(9, 7, 8)
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.outputColorSpace = THREE.SRGBColorSpace
    mount.appendChild(renderer.domElement)

    const positions = new Float32Array(cloud.positions.flat())
    const sourceColours = mode === 'rgb' ? cloud.rgb : cloud.semantic
    const colours = new Float32Array(sourceColours.flat())
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(colours, 3))
    geometry.computeBoundingSphere()
    const material = new THREE.PointsMaterial({ size: 0.035, vertexColors: true, sizeAttenuation: true })
    const points = new THREE.Points(geometry, material)
    if (showPoints) scene.add(points)

    const grid = new THREE.GridHelper(12, 24, '#c7c7cc', '#e5e5ea')
    grid.rotation.x = Math.PI / 2
    grid.position.z = -2.05
    scene.add(grid)

    const boxes = []
    const labels = []
    if (showBoxes) {
      nodes.forEach((node) => {
        const boxGeometry = new THREE.BoxGeometry(...node.bbox_dimensions)
        const edges = new THREE.EdgesGeometry(boxGeometry)
        const colour = node.synthetic ? '#ff3b30' : (node.id === selectedId ? '#0066cc' : '#1d1d1f')
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: node.id === selectedId ? 1 : 0.65 }))
        line.position.set(...node.centroid)
        line.userData.nodeId = node.id
        scene.add(line)
        boxes.push(line)
        if (showLabels) {
          const labelCanvas = document.createElement('canvas')
          labelCanvas.width = 512
          labelCanvas.height = 128
          const context = labelCanvas.getContext('2d')
          context.fillStyle = node.synthetic ? 'rgba(255,59,48,.90)' : 'rgba(29,29,31,.82)'
          context.beginPath()
          context.roundRect(8, 8, 496, 112, 24)
          context.fill()
          context.textAlign = 'center'
          context.fillStyle = '#ffffff'
          context.font = '700 38px system-ui, sans-serif'
          context.fillText(node.synthetic ? 'person' : node.label.replaceAll('_', ' '), 256, 57)
          context.fillStyle = 'rgba(255,255,255,.78)'
          context.font = '500 24px system-ui, sans-serif'
          context.fillText(`${node.bbox_dimensions.map((value) => Number(value).toFixed(2)).join(' × ')} m`, 256, 94)
          const texture = new THREE.CanvasTexture(labelCanvas)
          texture.colorSpace = THREE.SRGBColorSpace
          const spriteMaterial = new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false })
          const sprite = new THREE.Sprite(spriteMaterial)
          const width = Math.max(0.75, Math.min(1.8, Number(node.bbox_dimensions[0]) * 0.9))
          sprite.scale.set(width, width * 0.25, 1)
          sprite.position.set(...node.centroid)
          sprite.renderOrder = 10
          scene.add(sprite)
          labels.push({ sprite, texture, material: spriteMaterial })
        }
      })
    }

    let placementMarker = null
    if (placementMode && placementPoint) {
      const markerGeometry = new THREE.RingGeometry(0.10, 0.17, 32)
      const markerMaterial = new THREE.MeshBasicMaterial({ color: '#0066cc', side: THREE.DoubleSide, transparent: true, opacity: 0.95, depthTest: false })
      placementMarker = new THREE.Mesh(markerGeometry, markerMaterial)
      placementMarker.position.set(...placementPoint)
      placementMarker.renderOrder = 11
      scene.add(placementMarker)
    }

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.target.copy(geometry.boundingSphere?.center ?? new THREE.Vector3())
    controls.update()

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    let pointerStart = null
    const pointerDown = (event) => {
      pointerStart = { x: event.clientX, y: event.clientY }
    }
    const pick = (event) => {
      if (pointerStart && Math.hypot(event.clientX - pointerStart.x, event.clientY - pointerStart.y) > 5) {
        pointerStart = null
        return
      }
      pointerStart = null
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObjects(boxes)[0]
      if (!placementMode && hit?.object?.userData?.nodeId) onSelect?.(hit.object.userData.nodeId)
      if (placementMode && onPlace) {
        const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -placementZ)
        const position = new THREE.Vector3()
        if (raycaster.ray.intersectPlane(plane, position)) onPlace([position.x, position.y, placementZ])
      }
    }
    renderer.domElement.addEventListener('pointerdown', pointerDown)
    renderer.domElement.addEventListener('pointerup', pick)

    const resize = () => {
      const { clientWidth, clientHeight } = mount
      renderer.setSize(clientWidth, clientHeight, false)
      camera.aspect = clientWidth / Math.max(clientHeight, 1)
      camera.updateProjectionMatrix()
    }
    const observer = new ResizeObserver(resize)
    observer.observe(mount)
    resize()

    let frame
    const animate = () => {
      controls.update()
      renderer.render(scene, camera)
      frame = requestAnimationFrame(animate)
    }
    animate()
    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
      renderer.domElement.removeEventListener('pointerdown', pointerDown)
      renderer.domElement.removeEventListener('pointerup', pick)
      controls.dispose()
      geometry.dispose()
      material.dispose()
      boxes.forEach((box) => { box.geometry.dispose(); box.material.dispose() })
      labels.forEach(({ sprite, texture, material: labelMaterial }) => { scene.remove(sprite); texture.dispose(); labelMaterial.dispose() })
      if (placementMarker) { placementMarker.geometry.dispose(); placementMarker.material.dispose() }
      renderer.dispose()
      mount.replaceChildren()
    }
  }, [cloud, mode, nodes, onPlace, onSelect, placementMode, placementPoint, placementZ, selectedId, showBoxes, showLabels, showPoints])

  return <div className={`point-cloud ${placementMode ? 'placing' : ''}`} ref={mountRef} aria-label={placementMode ? 'Click the three-dimensional scene to place the worker' : 'Interactive three-dimensional point cloud'} />
}
