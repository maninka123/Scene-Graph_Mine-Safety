import { useMemo } from 'react'

export default function SceneGraph({ graph, colours, selectedId, onSelect }) {
  const layout = useMemo(() => {
    const anchor = graph.nodes.find((node) => node.label === 'equipment')
    const orbit = graph.nodes.filter((node) => node.id !== anchor?.id)
    const positions = anchor ? { [anchor.id]: { x: 410, y: 210 } } : {}
    orbit.forEach((node, index) => {
      const angle = -Math.PI / 2 + (index / Math.max(orbit.length, 1)) * Math.PI * 2
      positions[node.id] = { x: 410 + Math.cos(angle) * 285, y: 210 + Math.sin(angle) * 142 }
    })
    return positions
  }, [graph.nodes])
  const seen = new Set()
  const edges = graph.edges.filter((edge) => {
    const key = [edge.source, edge.target].sort().join('|')
    if (seen.has(key)) return false
    seen.add(key)
    return true
  })
  return (
    <svg className="scene-graph" viewBox="0 0 820 420" role="img" aria-label={`Scene graph with ${graph.nodes.length} nodes and ${edges.length} visible relations`}>
      <defs>
        <filter id="node-shadow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="5" stdDeviation="5" floodOpacity=".14" /></filter>
      </defs>
      {edges.map((edge) => {
        const source = layout[edge.source]
        const target = layout[edge.target]
        if (!source || !target) return null
        return <line key={`${edge.source}-${edge.target}`} x1={source.x} y1={source.y} x2={target.x} y2={target.y} className={edge.safety_flags?.length ? 'graph-edge danger' : 'graph-edge'} />
      })}
      {graph.nodes.map((node) => {
        const position = layout[node.id]
        const selected = node.id === selectedId
        return (
          <g key={node.id} transform={`translate(${position.x},${position.y})`} className="graph-node" onClick={() => onSelect(node.id)} role="button" tabIndex="0">
            <circle r={selected ? 25 : 21} fill={colours[node.label] || '#64d2ff'} stroke={selected ? '#0066cc' : '#fff'} strokeWidth={selected ? 5 : 3} filter="url(#node-shadow)" />
            <text y="38" textAnchor="middle">{node.label}</text>
            <text y="52" textAnchor="middle" className="node-id">{node.id}</text>
          </g>
        )
      })}
    </svg>
  )
}
