import { createElement, useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, AlertTriangle, ChevronRight, CircleGauge, Clock3, GitBranch, HardHat, Info, Play, Plus, RotateCcw, ScanLine, ShieldCheck, Sparkles } from 'lucide-react'
import PointCloud from './components/PointCloud.jsx'
import SceneGraph from './components/SceneGraph.jsx'

const API = import.meta.env.VITE_API_URL || ''
const tabs = [
  { id: 'perception', label: 'Perception', icon: ScanLine },
  { id: 'graph', label: 'Scene graph', icon: GitBranch },
  { id: 'reasoning', label: 'Reasoning', icon: ShieldCheck },
  { id: 'performance', label: 'Performance', icon: CircleGauge },
]

async function loadDemo() {
  try {
    const response = await fetch(`${API}/api/demo`)
    if (!response.ok) throw new Error('API unavailable')
    return { data: await response.json(), api: true }
  } catch {
    const response = await fetch('./demo.json')
    if (!response.ok) throw new Error('Demo bundle could not be loaded')
    return { data: await response.json(), api: false }
  }
}

function Stat({ label, value, suffix }) {
  return <div className="stat"><span>{label}</span><strong>{value}{suffix && <small>{suffix}</small>}</strong></div>
}

function Pipeline({ stages, activeStage, runningStage, setActiveStage }) {
  return (
    <div className="pipeline" aria-label="Processing pipeline">
      {stages.map((stage, index) => {
        const active = stage.id === activeStage
        const running = stage.id === runningStage
        return <div className="pipeline-wrap" key={stage.id}>
          <button className={`pipeline-stage ${active ? 'active' : ''} ${running ? 'running' : ''}`} onClick={() => setActiveStage(stage.id)}>
            <span className="stage-index">{running ? <span className="spinner" /> : index + 1}</span>
            <span><b>{stage.label}</b><small>{stage.detail}</small></span>
            <em className={`source-tag ${stage.kind}`}>{stage.kind}</em>
          </button>
          {index < stages.length - 1 && <ChevronRight size={15} className="pipeline-arrow" />}
        </div>
      })}
    </div>
  )
}

function Segmented({ value, onChange, options, label }) {
  return <div className="segmented" aria-label={label}>{options.map((option) => <button key={option.value} className={value === option.value ? 'selected' : ''} onClick={() => onChange(option.value)}>{option.label}</button>)}</div>
}

function App() {
  const [payload, setPayload] = useState(null)
  const [error, setError] = useState('')
  const [apiConnected, setApiConnected] = useState(false)
  const [tab, setTab] = useState('perception')
  const [colourMode, setColourMode] = useState('semantic')
  const [showBoxes, setShowBoxes] = useState(true)
  const [selectedId, setSelectedId] = useState(null)
  const [edgeDistance, setEdgeDistance] = useState(2.5)
  const [nodes, setNodes] = useState([])
  const [graph, setGraph] = useState(null)
  const [alerts, setAlerts] = useState([])
  const [runningStage, setRunningStage] = useState(null)
  const [activeStage, setActiveStage] = useState('minkunet')
  const [liveTimings, setLiveTimings] = useState(null)

  useEffect(() => {
    loadDemo().then(({ data, api }) => {
      setPayload(data); setApiConnected(api); setNodes(data.nodes); setGraph(data.graph); setAlerts(data.alerts); setSelectedId(data.nodes[1]?.id)
    }).catch((reason) => setError(reason.message))
  }, [])

  const runAnalysis = useCallback(async (nextNodes = nodes, threshold = edgeDistance) => {
    if (!payload) return
    const animation = ['pcd', 'voxelise', 'minkunet', 'instances', 'graph', 'rules']
    for (const stage of animation) {
      setRunningStage(stage); setActiveStage(stage)
      await new Promise((resolve) => setTimeout(resolve, window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 20 : 180))
    }
    if (apiConnected) {
      const response = await fetch(`${API}/api/analyse`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ nodes: nextNodes, edge_distance_m: threshold }) })
      if (!response.ok) throw new Error('Live Python analysis failed')
      const result = await response.json()
      setGraph(result.graph); setAlerts(result.alerts); setLiveTimings(result.live_timings)
    } else {
      setGraph({ ...payload.graph, nodes: nextNodes })
    }
    setRunningStage(null)
  }, [apiConnected, edgeDistance, nodes, payload])

  useEffect(() => {
    if (!payload || !apiConnected) return
    const timer = setTimeout(() => runAnalysis(nodes, edgeDistance).catch((reason) => setError(reason.message)), 280)
    return () => clearTimeout(timer)
  }, [edgeDistance]) // eslint-disable-line react-hooks/exhaustive-deps

  const addWorker = async () => {
    const equipment = nodes.find((node) => node.label === 'equipment')
    const worker = {
      id: `personnel-test-${nodes.filter((node) => node.synthetic).length + 1}`,
      label: 'personnel', centroid: [equipment.centroid[0] + 0.8, equipment.centroid[1], equipment.centroid[2]],
      bbox_dimensions: [0.5, 0.5, 1.75], orientation: [1, 0, 0], volume_m3: 0.44, voxel_count: 1,
      confidence: 1, entropy: 0, is_anomaly: false, active: true, velocity: [0, 0, 0], movement_state: 'stationary', synthetic: true,
    }
    const next = [...nodes, worker]
    setNodes(next); setSelectedId(worker.id); setTab('graph')
    await runAnalysis(next, edgeDistance)
  }

  const reset = () => {
    setNodes(payload.nodes); setGraph(payload.graph); setAlerts(payload.alerts); setSelectedId(payload.nodes[1]?.id); setEdgeDistance(2.5); setLiveTimings(null)
  }

  const selected = nodes.find((node) => node.id === selectedId)
  const maxTiming = useMemo(() => payload ? Math.max(...payload.timings.filter((item) => item.mean_ms).map((item) => item.mean_ms)) : 1, [payload])

  if (error) return <main className="fatal"><AlertTriangle /><h1>MineGraph Studio could not start</h1><p>{error}</p></main>
  if (!payload || !graph) return <main className="loading"><span className="loader" /><p>Preparing simulator frame…</p></main>

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><img src="./app-icon.svg" alt="" /><span><b>MineGraph</b> Studio</span></div>
        <nav>{tabs.map(({ id, label, icon }) => <button key={id} className={tab === id ? 'active' : ''} onClick={() => setTab(id)}>{createElement(icon, { size: 16 })}{label}</button>)}</nav>
        <div className={`connection ${apiConnected ? 'online' : 'static'}`}><i />{apiConnected ? 'Python API live' : 'Static demo'}</div>
      </header>

      <main className="workspace">
        <section className="hero">
          <div>
            <div className="eyebrow"><span>RESEARCH DEMONSTRATOR</span><span className="wip">Work in progress</span></div>
            <h1>From raw point cloud<br />to explainable safety context.</h1>
            <p>Explore a recorded simulator frame through the paper’s perception, scene-graph, and deterministic reasoning pipeline.</p>
          </div>
          <div className="hero-actions">
            <button className="primary" onClick={() => runAnalysis().catch((reason) => setError(reason.message))} disabled={Boolean(runningStage)}><Play size={16} fill="currentColor" />{runningStage ? 'Running pipeline…' : 'Run pipeline'}</button>
            <button className="secondary" onClick={addWorker}><Plus size={16} />Add test worker</button>
          </div>
        </section>

        <aside className="notice"><Info size={17} /><span><b>Demonstration only.</b> Recorded MinkUNet outputs are shown for this included scan; graph construction and rule checks run live when the Python API is connected. This is not connected to a simulator, sensors, alarms, or operational controls.</span></aside>

        <Pipeline stages={payload.pipeline} activeStage={activeStage} runningStage={runningStage} setActiveStage={setActiveStage} />

        <section className="metrics-strip">
          <Stat label="Input points" value={payload.demo.point_count.toLocaleString()} />
          <Stat label="Sparse voxels" value={payload.demo.voxel_count.toLocaleString()} />
          <Stat label="Detected objects" value={nodes.length} />
          <Stat label="Directed relations" value={graph.edges.length} />
          <Stat label="Rule alerts" value={alerts.length} />
          <Stat label="Reference GPU" value="RTX 4080" />
        </section>

        {tab === 'perception' && <section className="content-grid">
          <article className="card visual-card">
            <div className="card-head"><div><span className="overline">3D PERCEPTION</span><h2>{payload.demo.filename}</h2></div><div className="visual-controls"><Segmented label="Point colouring" value={colourMode} onChange={setColourMode} options={[{ value: 'semantic', label: 'Semantic' }, { value: 'rgb', label: 'RGB' }]} /><label className="switch"><input type="checkbox" checked={showBoxes} onChange={(event) => setShowBoxes(event.target.checked)} /><span />Boxes</label></div></div>
            <PointCloud cloud={payload.point_cloud} nodes={nodes} mode={colourMode} showBoxes={showBoxes} selectedId={selectedId} onSelect={setSelectedId} />
            <div className="canvas-hint">Drag to orbit · Scroll to zoom · Select a detection box</div>
          </article>
          <aside className="side-stack">
            <article className="card inspector"><div className="card-head"><div><span className="overline">DETECTION</span><h2>{selected ? selected.label : 'Select an object'}</h2></div>{selected?.synthetic && <span className="synthetic-badge">Synthetic</span>}</div>
              {selected ? <><div className="object-id">{selected.id}</div><dl><div><dt>Confidence</dt><dd>{(selected.confidence * 100).toFixed(0)}%</dd></div><div><dt>Points</dt><dd>{selected.voxel_count.toLocaleString()}</dd></div><div><dt>Volume</dt><dd>{selected.volume_m3.toFixed(2)} m³</dd></div><div><dt>Centroid</dt><dd>{selected.centroid.map((v) => v.toFixed(2)).join(', ')}</dd></div></dl></> : <p>Click a box in the 3D view.</p>}
            </article>
            <article className="card"><span className="overline">CLASS DISTRIBUTION</span><div className="distribution">{Object.entries(payload.demo.class_counts).filter(([, count]) => count > 0).map(([name, count]) => <div key={name}><div><span><i style={{ background: payload.class_colours[name] }} />{name}</span><b>{count.toLocaleString()}</b></div><span className="bar"><i style={{ width: `${count / payload.demo.point_count * 100}%`, background: payload.class_colours[name] }} /></span></div>)}</div></article>
          </aside>
        </section>}

        {tab === 'graph' && <section className="content-grid">
          <article className="card visual-card"><div className="card-head"><div><span className="overline">OBJECT-RELATION GRAPH</span><h2>Interactive scene graph</h2></div><span className="graph-count">{graph.nodes.length} nodes · {graph.edges.length} directed edges</span></div><SceneGraph graph={graph} colours={payload.class_colours} selectedId={selectedId} onSelect={setSelectedId} /></article>
          <aside className="side-stack"><article className="card"><div className="card-head"><div><span className="overline">GRAPH CONTROLS</span><h2>Relation radius</h2></div><b>{edgeDistance.toFixed(1)} m</b></div><input className="range" type="range" min="0.5" max="8" step="0.25" value={edgeDistance} onChange={(event) => setEdgeDistance(Number(event.target.value))} /><div className="range-labels"><span>Local</span><span>Paper maximum 8 m</span></div><p className="muted">The graph is rebuilt in Python as this radius changes. Reciprocal directed relations are retained by the paper-aligned graph contract.</p></article><article className="card"><div className="card-head"><div><span className="overline">SCENE EDITOR</span><h2>Test a safety case</h2></div><HardHat size={22} /></div><p className="muted">Add a clearly marked synthetic worker beside the detected equipment to exercise proximity evidence without altering the recorded perception result.</p><div className="button-row"><button className="primary small" onClick={addWorker}><Plus size={15} />Add worker</button><button className="secondary small" onClick={reset}><RotateCcw size={15} />Reset</button></div></article></aside>
        </section>}

        {tab === 'reasoning' && <section className="reasoning-grid">
          <article className={`card safety-summary ${alerts.length ? 'warning' : 'safe'}`}><div className="status-icon">{alerts.length ? <AlertTriangle /> : <ShieldCheck />}</div><div><span className="overline">DETERMINISTIC LAYER</span><h2>{alerts.length ? `${alerts.length} rule finding${alerts.length === 1 ? '' : 's'}` : 'No rule findings in recorded frame'}</h2><p>{alerts.length ? 'Review the grounded evidence below. Synthetic scenario objects are explicitly labelled.' : 'The recorded scan contains no personnel detection, so person–equipment rules do not fire.'}</p></div></article>
          <div className="findings">{alerts.length ? alerts.map((alert, index) => <article className="card finding" key={`${alert.rule}-${index}`}><div className="finding-head"><span className={`severity ${alert.severity}`}>{alert.severity}</span><b>{alert.rule.replaceAll('_', ' ')}</b></div><p>{alert.message}</p><div className="evidence">{Object.entries(alert.evidence).map(([key, value]) => <span key={key}><small>{key.replaceAll('_', ' ')}</small><b>{typeof value === 'number' ? value.toFixed(3) : value}</b></span>)}</div><div className="object-links">{alert.object_ids.map((id) => <button key={id} onClick={() => { setSelectedId(id); setTab('graph') }}>{id}</button>)}</div></article>) : <article className="card empty-state"><ShieldCheck /><h3>Safe under current deterministic rules</h3><p>Add a test worker to see object-grounded proximity evidence.</p><button className="primary small" onClick={addWorker}><Plus size={15} />Add test worker</button></article>}</div>
          <article className="card qwen-card"><div><Sparkles size={20} /><span><b>Qwen contextual reasoning</b><small>Optional local model · not run in this browser session</small></span></div><span className="source-tag optional">optional</span><p>The production path receives the bounded scene graph—not raw point clouds—and must return schema-validated, object-grounded JSON. Recorded reference latency is shown under Performance.</p></article>
        </section>}

        {tab === 'performance' && <section className="performance-grid">
          <article className="card timing-card"><div className="card-head"><div><span className="overline">REFERENCE EXECUTION TIME</span><h2>Measured pipeline stages</h2></div><span className="recorded-pill"><Activity size={14} />Recorded · 7 runs</span></div><div className="timings">{payload.timings.map((timing) => <div className="timing-row" key={timing.id}><span>{timing.label}<small>{timing.source}</small></span><div className="timing-track"><i style={{ width: `${Math.max(1, Math.sqrt(timing.mean_ms / maxTiming) * 100)}%` }} /></div><b>{timing.mean_ms < 1 ? timing.mean_ms.toFixed(3) : timing.mean_ms.toFixed(1)} ms</b></div>)}</div><p className="footnote">Bars use a square-root scale so sub-millisecond graph stages remain visible beside Qwen generation. Values are recorded measurements, not estimates.</p></article>
          <aside className="side-stack"><article className="card"><span className="overline">BENCHMARK SYSTEM</span><h2>{payload.timing_environment.gpu}</h2><dl className="system-list"><div><dt>CUDA</dt><dd>{payload.timing_environment.cuda_available ? 'Available' : 'Unavailable'}</dd></div><div><dt>Frame repeats</dt><dd>{payload.timing_environment.repeats}</dd></div><div><dt>Context model</dt><dd>{payload.timing_environment.model}</dd></div><div><dt>Measured</dt><dd>{payload.timing_environment.generated_at.slice(0, 10)}</dd></div></dl></article><article className="card live-card"><span className="overline">THIS SESSION</span><h2>Live post-processing</h2>{liveTimings ? <><div className="live-number">{liveTimings.total_postprocess_ms.toFixed(3)} <small>ms</small></div><p>Python graph construction and deterministic rule evaluation on this machine.</p></> : <><Clock3 /><p>Run the pipeline to measure the lightweight graph and rule stages on this machine.</p></>}</article></aside>
        </section>}

        <footer><span>MineGraph Studio · Paper-aligned research interface</span><span>One included simulator frame · No live control connection</span></footer>
      </main>
    </div>
  )
}

export default App
