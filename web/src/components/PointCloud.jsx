import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export default function PointCloud({ cloud, nodes, mode, showBoxes, selectedId, onSelect }) {
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
    scene.add(points)

    const grid = new THREE.GridHelper(12, 24, '#c7c7cc', '#e5e5ea')
    grid.rotation.x = Math.PI / 2
    grid.position.z = -2.05
    scene.add(grid)

    const boxes = []
    if (showBoxes) {
      nodes.forEach((node) => {
        const boxGeometry = new THREE.BoxGeometry(...node.bbox_dimensions)
        const edges = new THREE.EdgesGeometry(boxGeometry)
        const colour = node.id === selectedId ? '#0066cc' : (node.synthetic ? '#ff3b30' : '#1d1d1f')
        const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({ color: colour, transparent: true, opacity: node.id === selectedId ? 1 : 0.65 }))
        line.position.set(...node.centroid)
        line.userData.nodeId = node.id
        scene.add(line)
        boxes.push(line)
      })
    }

    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.target.copy(geometry.boundingSphere?.center ?? new THREE.Vector3())
    controls.update()

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const pick = (event) => {
      const rect = renderer.domElement.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, camera)
      const hit = raycaster.intersectObjects(boxes)[0]
      if (hit?.object?.userData?.nodeId) onSelect(hit.object.userData.nodeId)
    }
    renderer.domElement.addEventListener('pointerdown', pick)

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
      renderer.domElement.removeEventListener('pointerdown', pick)
      controls.dispose()
      geometry.dispose()
      material.dispose()
      boxes.forEach((box) => { box.geometry.dispose(); box.material.dispose() })
      renderer.dispose()
      mount.replaceChildren()
    }
  }, [cloud, mode, nodes, onSelect, selectedId, showBoxes])

  return <div className="point-cloud" ref={mountRef} aria-label="Interactive three-dimensional point cloud" />
}
