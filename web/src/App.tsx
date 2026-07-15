import { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from "react"
import {
  Activity,
  AlertTriangle,
  Check,
  ChevronRight,
  CircleStop,
  Clock3,
  Database,
  Download,
  Gauge,
  KeyRound,
  Moon,
  Play,
  RefreshCw,
  Server,
  Settings2,
  Sun,
  TerminalSquare,
  Zap,
} from "lucide-react"
import { toast } from "sonner"
import { api, type DatasetOption, type Options, type Run, type RunConfig } from "./api"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Switch } from "@/components/ui/switch"

const ACTIVE_STATES = new Set(["queued", "running", "cancelling"])

const DEFAULT_CONFIG: RunConfig = {
  dataset_source: "huggingface",
  dataset_path: "bettercallaaryan/nc_agent_clips_openasr",
  dataset: "default",
  split: "test",
  model_name: "deepgram/nova-3",
  language: "en",
  max_samples: null,
  max_workers: 4,
  use_url: false,
  streaming: false,
  prompt: null,
  audio_preprocessor: "arctan",
  vad_position: "post",
  arctan_chunk_ms: 10,
  audio_preprocessor_batch_size: 1,
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  )
}

function ToggleRow({ label, description, checked, disabled, onCheckedChange }: { label: string; description: string; checked: boolean; disabled?: boolean; onCheckedChange: (checked: boolean) => void }) {
  return (
    <div className="toggle-row">
      <div>
        <div className="text-sm font-medium">{label}</div>
        <div className="mt-0.5 text-xs leading-relaxed text-muted-foreground">{description}</div>
      </div>
      <Switch checked={checked} disabled={disabled} onCheckedChange={onCheckedChange} aria-label={label} />
    </div>
  )
}

function StatusBadge({ status }: { status: Run["status"] }) {
  const tone = status === "completed" ? "success" : status === "failed" || status === "interrupted" ? "danger" : status === "cancelled" ? "muted" : "active"
  return <Badge className={`status-badge status-${tone}`}><span className="status-dot" />{status}</Badge>
}

function pipelineLabel(run: Run) {
  const pieces = []
  if (run.config.vad_position === "pre") pieces.push("VAD")
  if (run.config.audio_preprocessor !== "none") pieces.push(run.config.audio_preprocessor.replace("ai_coustics_vfl_2_1", "ai-coustics"))
  if (run.config.vad_position === "post") pieces.push("VAD")
  return pieces.length ? pieces.join(" → ") : "Raw audio"
}

function configuredModelLabel(modelName: string) {
  return modelName === "assembly/universal-stt" ? "Assembly Universal STT" : modelName
}

function observedCounts(run: Run, key: "actual_models" | "detected_languages") {
  return run.progress?.[key] || run.summary?.[key] || {}
}

function countLabel(counts: Record<string, number>) {
  const entries = Object.entries(counts)
  if (!entries.length) return "Waiting for provider response…"
  return entries.map(([name, count]) => `${name} · ${count}`).join("  |  ")
}

function elapsed(run: Run) {
  const start = run.started_at || run.created_at
  const end = run.finished_at || new Date().toISOString()
  const seconds = Math.max(0, Math.round((new Date(end).getTime() - new Date(start).getTime()) / 1000))
  if (seconds < 60) return `${seconds}s`
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

function RunComposer({ options, activeCount, onCreated }: { options: Options; activeCount: number; onCreated: (run: Run) => void }) {
  const [config, setConfig] = useState<RunConfig>(DEFAULT_CONFIG)
  const [submitting, setSubmitting] = useState(false)
  const [datasets, setDatasets] = useState<DatasetOption[]>([])
  const [datasetsLoading, setDatasetsLoading] = useState(false)
  const [datasetError, setDatasetError] = useState<string | null>(null)
  const datasetRequest = useRef(0)

  const modelOptions = useMemo(() => options.providers.flatMap((provider) => provider.models.map((model) => ({ value: `${provider.prefix}/${model}`, label: `${provider.label} · ${model}`, configured: provider.configured }))), [options])
  const provider = options.providers.find((item) => config.model_name.startsWith(`${item.prefix}/`))
  const isSarvam = config.model_name.startsWith("sarvam/")
  const forcedStreaming = config.model_name === "soniox/stt-rt-v5"
  const effectiveStreaming = config.streaming || forcedStreaming
  const modelVariant = config.model_name.split("/", 2)[1]
  const languageOptions = useMemo(
    () => provider?.language_options[modelVariant]?.[effectiveStreaming ? "streaming" : "batch"] ?? [],
    [effectiveStreaming, modelVariant, provider],
  )
  const unsupportedMode = languageOptions.length === 0
  const providerRequiresLocal = isSarvam
  const localTransforms = config.dataset_source === "local" || config.streaming || forcedStreaming || providerRequiresLocal || config.audio_preprocessor !== "none" || config.vad_position !== "none"
  const preprocessorCredentials: Record<string, string[]> = {
    arctan: ["ARCTAN_SDK_KEY"],
    ai_coustics_vfl_2_1: ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"],
    krisp_bvc: ["LIVEKIT_URL", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"],
  }
  const missingPreprocessorCredential = (preprocessorCredentials[config.audio_preprocessor] || []).find((name) => !options.credentials[name])
  const missingCredential = !provider?.configured || Boolean(missingPreprocessorCredential)

  useEffect(() => {
    if (!languageOptions.length) return
    if (languageOptions.some((option) => option.code === config.language)) return
    setConfig((current) => ({ ...current, language: languageOptions[0].code }))
  }, [config.language, languageOptions])

  function update<K extends keyof RunConfig>(key: K, value: RunConfig[K]) {
    setConfig((current) => ({ ...current, [key]: value }))
  }

  const selectedDataset = datasets.find((item) => item.dataset_source === config.dataset_source && item.dataset_path === config.dataset_path && item.dataset === config.dataset)
  const invalidDatasets = datasets.filter((item) => !item.valid)

  const loadDatasets = useCallback(async (source: RunConfig["dataset_source"]) => {
    const requestId = ++datasetRequest.current
    setDatasetsLoading(true)
    setDatasetError(null)
    try {
      const result = await api.datasets(source)
      if (requestId !== datasetRequest.current) return
      setDatasets(result.datasets)
      const first = result.datasets.find((item) => item.valid && item.splits.length > 0)
      setConfig((current) => {
        if (current.dataset_source !== source) return current
        return {
          ...current,
          dataset_path: first?.dataset_path ?? "",
          dataset: first?.dataset ?? "",
          split: first?.splits[0] ?? "",
          use_url: source === "local" ? false : current.use_url,
        }
      })
    } catch (error) {
      if (requestId !== datasetRequest.current) return
      setDatasets([])
      setDatasetError(error instanceof Error ? error.message : "Could not load datasets")
      setConfig((current) => current.dataset_source === source ? { ...current, dataset_path: "", dataset: "", split: "" } : current)
    } finally {
      if (requestId === datasetRequest.current) setDatasetsLoading(false)
    }
  }, [])

  useEffect(() => {
    void loadDatasets(config.dataset_source)
  }, [config.dataset_source, loadDatasets])

  function selectDataset(id: string) {
    const selected = datasets.find((item) => item.id === id && item.valid)
    if (!selected) return
    setConfig((current) => ({
      ...current,
      dataset_path: selected.dataset_path,
      dataset: selected.dataset,
      split: selected.splits[0],
    }))
  }
  async function submit(event: FormEvent) {
    event.preventDefault()
    setSubmitting(true)
    try {
      const run = await api.createRun(config)
      toast.success("Evaluation started", { description: `${configuredModelLabel(run.config.model_name)} · ${run.id}` })
      onCreated(run)
    } catch (error) {
      toast.error("Could not start evaluation", { description: error instanceof Error ? error.message : "Unknown error" })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form className="composer-card" onSubmit={submit}>
      <div className="composer-heading">
        <div>
          <div className="eyebrow"><Zap className="size-3.5" />New evaluation</div>
          <h1>Configure a run</h1>
        </div>
        <Badge className="active-count">{activeCount} active</Badge>
      </div>

      {activeCount > 0 && (
        <div className="warning-strip"><AlertTriangle className="size-4" /><span>Runs start immediately with no global cap. Check workers and provider usage.</span></div>
      )}

      <div className="form-section">
        <div className="section-label"><Database className="size-4" />Dataset</div>
        <Field label="Source">
          <Select value={config.dataset_source} onValueChange={(value) => setConfig((current) => ({ ...current, dataset_source: value as RunConfig["dataset_source"], dataset_path: "", dataset: "", split: "", use_url: value === "local" ? false : current.use_url }))}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>{options.dataset_sources.map((source) => <SelectItem value={source.id} key={source.id}>{source.label} · {source.description}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <Field label="Dataset">
          <div className="joined-control">
            <Select disabled={datasetsLoading || datasets.length === 0} value={selectedDataset?.id ?? ""} onValueChange={selectDataset}>
              <SelectTrigger><SelectValue placeholder={datasetsLoading ? "Loading datasets…" : "Select a compatible dataset"} /></SelectTrigger>
              <SelectContent>{datasets.map((item) => <SelectItem value={item.id} key={item.id} disabled={!item.valid} title={item.error ?? undefined}>{item.label}{item.valid ? "" : " · incompatible"}</SelectItem>)}</SelectContent>
            </Select>
            <Button type="button" variant="outline" onClick={() => void loadDatasets(config.dataset_source)} disabled={datasetsLoading} aria-label="Refresh datasets">
              <RefreshCw className={`size-4 ${datasetsLoading ? "spin" : ""}`} />
            </Button>
          </div>
        </Field>
        {datasetError && <div className="dataset-load-error" role="alert"><AlertTriangle className="size-3.5" />{datasetError}</div>}
        {!datasetsLoading && !datasetError && datasets.length === 0 && <div className="dataset-empty">This source exposes no dataset candidates.</div>}
        {selectedDataset && <div className="dataset-note"><Check className="size-3.5" />{selectedDataset.features.join(", ")} · {selectedDataset.splits.length} split{selectedDataset.splits.length === 1 ? "" : "s"}</div>}
        {invalidDatasets.length > 0 && <div className="dataset-validation-errors" aria-label="Incompatible datasets">{invalidDatasets.map((item) => <div key={item.id}><strong>{item.label}</strong><span>{item.error}</span></div>)}</div>}
        <Field label="Split">
          <Select disabled={!selectedDataset} value={config.split} onValueChange={(value) => update("split", value)}>
            <SelectTrigger><SelectValue placeholder="Select a split" /></SelectTrigger>
            <SelectContent>{selectedDataset?.splits.map((split) => <SelectItem value={split} key={split}>{split}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
      </div>

      <div className="form-section">
        <div className="section-label"><Activity className="size-4" />Recognition</div>
        <Field label="Provider and model">
          <Select value={config.model_name} onValueChange={(value) => { update("model_name", value); if (value === "soniox/stt-rt-v5" || value.startsWith("sarvam/")) update("use_url", false) }}>
            <SelectTrigger><SelectValue /></SelectTrigger>
            <SelectContent>{modelOptions.map((item) => <SelectItem value={item.value} key={item.value}>{item.label}{item.configured ? "" : " · missing key"}</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <Field label="Language" hint={unsupportedMode ? `${effectiveStreaming ? "Streaming" : "Batch"} evaluation is not supported for this model` : config.language === "unknown" ? "The provider detects the spoken language" : "Only languages supported by this model and mode are shown"}>
          <Select disabled={unsupportedMode} value={unsupportedMode ? "" : config.language} onValueChange={(value) => update("language", value)}>
            <SelectTrigger><SelectValue placeholder="No supported language options" /></SelectTrigger>
            <SelectContent>{languageOptions.map((option) => <SelectItem value={option.code} key={option.code}>{option.label} ({option.code})</SelectItem>)}</SelectContent>
          </Select>
        </Field>
        <div className="form-grid-2">
          <Field label="Workers" hint="Per-run concurrency"><Input type="number" min={1} max={300} value={config.max_workers} onChange={(event) => update("max_workers", Number(event.target.value))} /></Field>
          <Field label="Sample limit" hint="Blank runs full split"><Input type="number" min={1} value={config.max_samples ?? ""} placeholder="Full" onChange={(event) => update("max_samples", event.target.value ? Number(event.target.value) : null)} /></Field>
        </div>
      </div>

      <div className="form-section">
        <div className="section-label"><Settings2 className="size-4" />Audio pipeline</div>
        <div className="form-grid-2">
          <Field label="Preprocessor">
            <Select value={config.audio_preprocessor} onValueChange={(value) => { update("audio_preprocessor", value); if (value !== "none") update("use_url", false) }}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>{options.preprocessors.map((item) => <SelectItem value={item} key={item}>{item.replaceAll("_", " ")}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
          <Field label="VAD position">
            <Select value={config.vad_position} onValueChange={(value) => { update("vad_position", value); if (value !== "none") update("use_url", false) }}>
              <SelectTrigger><SelectValue /></SelectTrigger>
              <SelectContent>{options.vad_positions.map((item) => <SelectItem value={item} key={item}>{item === "none" ? "No VAD" : `${item}-processor`}</SelectItem>)}</SelectContent>
            </Select>
          </Field>
        </div>
        {config.vad_position !== "none" && (
          <div className="pipeline-preview"><span>Pipeline</span><strong>{config.vad_position === "pre" ? "Audio → VAD → Processor → ASR" : "Audio → Processor → VAD → ASR"}</strong></div>
        )}
        <ToggleRow label="Streaming endpoint" description="Send paced PCM chunks to supported providers." checked={config.streaming} onCheckedChange={(checked) => { update("streaming", checked); if (checked) update("use_url", false) }} />
        <ToggleRow label="Remote URL mode" description={config.dataset_source === "local" ? "Unavailable for local datasets." : localTransforms ? "Unavailable while streaming, preprocessing, VAD, or a real-time-only model is enabled." : "Let supported providers fetch dataset audio URLs directly."} checked={config.use_url} disabled={localTransforms} onCheckedChange={(checked) => update("use_url", checked)} />
      </div>

      <details className="advanced-panel">
        <summary><span><Settings2 className="size-4" />Advanced</span><ChevronRight className="size-4 chevron" /></summary>
        <div className="advanced-content">
          <div className="form-grid-2">
            <Field label="Arctan chunk (ms)"><Input type="number" min={1} value={config.arctan_chunk_ms} onChange={(event) => update("arctan_chunk_ms", Number(event.target.value))} /></Field>
            <Field label="Preprocess batch"><Input type="number" min={1} value={config.audio_preprocessor_batch_size} onChange={(event) => update("audio_preprocessor_batch_size", Number(event.target.value))} /></Field>
          </div>
          <Field label="Prompt"><textarea className="textarea" value={config.prompt ?? ""} onChange={(event) => update("prompt", event.target.value || null)} placeholder="Optional provider instruction" /></Field>
        </div>
      </details>

      {missingCredential && <div className="credential-error"><KeyRound className="size-4" />{missingPreprocessorCredential ? `${missingPreprocessorCredential} is missing on the server.` : `${provider?.label || "Provider"} credentials are missing on the server.`}</div>}

      <Button type="submit" className="h-11 w-full" disabled={submitting || datasetsLoading || !selectedDataset?.valid || missingCredential || unsupportedMode}>
        {submitting ? <RefreshCw className="size-4 spin" /> : <Play className="size-4 fill-current" />}
        {submitting ? "Starting…" : "Run evaluation"}
      </Button>
    </form>
  )
}

function RunTable({ runs, onSelect }: { runs: Run[]; onSelect: (run: Run) => void }) {
  if (!runs.length) {
    return <div className="empty-state"><div className="empty-icon"><TerminalSquare /></div><h3>No evaluations yet</h3><p>Configure your first run. Logs, metrics, and artifacts will appear here.</p></div>
  }
  return (
    <div className="table-shell">
      <table>
        <thead><tr><th>Status</th><th>Model</th><th>Dataset</th><th>Pipeline</th><th>WER</th><th>RTFx</th><th>Duration</th><th /></tr></thead>
        <tbody>{runs.map((run) => (
          <tr key={run.id} onClick={() => onSelect(run)} tabIndex={0} onKeyDown={(event) => { if (event.key === "Enter") onSelect(run) }}>
            <td><StatusBadge status={run.status} /></td>
            <td><div className="cell-primary">{configuredModelLabel(run.config.model_name)}</div><div className="cell-secondary">{Object.keys(observedCounts(run, "actual_models")).length ? countLabel(observedCounts(run, "actual_models")) : run.id}</div></td>
            <td><div className="cell-primary truncate-cell">{run.config.dataset_path}</div><div className="cell-secondary">{run.config.dataset} · {run.config.split}</div></td>
            <td><span className="pipeline-cell">{pipelineLabel(run)}</span></td>
            <td className="metric-cell">{run.summary?.wer_percent != null ? `${run.summary.wer_percent}%` : "—"}</td>
            <td className="metric-cell">{run.summary?.rtfx ?? "—"}</td>
            <td className="text-muted-foreground">{elapsed(run)}</td>
            <td><ChevronRight className="size-4 text-muted-foreground" /></td>
          </tr>
        ))}</tbody>
      </table>
    </div>
  )
}

function RunDetail({ selected, open, onOpenChange, onUpdated }: { selected: Run | null; open: boolean; onOpenChange: (open: boolean) => void; onUpdated: (run: Run) => void }) {
  const [run, setRun] = useState<Run | null>(selected)
  const [logs, setLogs] = useState("")
  const selectedRef = useRef(selected)
  const onUpdatedRef = useRef(onUpdated)
  selectedRef.current = selected
  onUpdatedRef.current = onUpdated
  const selectedId = selected?.id

  useEffect(() => {
    setRun(selectedRef.current)
    setLogs("")
    if (!selectedId || !open) return
    const events = new EventSource(`/api/runs/${selectedId}/events`)
    events.addEventListener("log", (event) => {
      const data = JSON.parse((event as MessageEvent).data)
      setLogs((current) => current + data.chunk)
    })
    events.addEventListener("status", (event) => {
      const updated = JSON.parse((event as MessageEvent).data) as Run
      setRun(updated)
      onUpdatedRef.current(updated)
      if (ACTIVE_STATES.has(updated.status)) return
      events.close()
    })
    events.addEventListener("progress", (event) => {
      const updated = JSON.parse((event as MessageEvent).data) as Run
      setRun(updated)
      onUpdatedRef.current(updated)
    })
    return () => events.close()
  }, [selectedId, open])

  async function cancel() {
    if (!run) return
    try {
      const updated = await api.cancelRun(run.id)
      setRun(updated)
      onUpdated(updated)
      toast.success("Cancellation requested")
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not cancel run")
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent>
        {run && <>
          <SheetHeader>
            <div className="mb-3 flex items-center gap-2"><StatusBadge status={run.status} /><span className="font-mono text-xs text-muted-foreground">{run.id}</span></div>
            <SheetTitle>{configuredModelLabel(run.config.model_name)}</SheetTitle>
            <SheetDescription>{run.config.dataset_path} · {run.config.dataset}/{run.config.split}</SheetDescription>
          </SheetHeader>
          <div className="detail-scroll">
            <div className="metric-grid">
              <div className="metric-card"><span>WER</span><strong>{run.summary?.wer_percent != null ? `${run.summary.wer_percent}%` : "—"}</strong></div>
              <div className="metric-card"><span>RTFx</span><strong>{run.summary?.rtfx ?? "—"}</strong></div>
              <div className="metric-card"><span>Samples</span><strong>{run.summary?.num_samples ?? "—"}</strong></div>
              <div className="metric-card"><span>Duration</span><strong>{elapsed(run)}</strong></div>
            </div>

            <section className="detail-section">
              <div className="detail-section-title"><Gauge className="size-4" />Run configuration</div>
              <dl className="config-list">
                <div><dt>Pipeline</dt><dd>{pipelineLabel(run)}</dd></div>
                <div><dt>Workers</dt><dd>{run.config.max_workers}</dd></div>
                <div><dt>Mode</dt><dd>{run.config.streaming ? "Streaming" : "Static"}</dd></div>
                <div><dt>Language</dt><dd>{run.config.language}</dd></div>
                <div><dt>Using ASR</dt><dd>{countLabel(observedCounts(run, "actual_models"))}</dd></div>
                {Object.keys(observedCounts(run, "detected_languages")).length > 0 && <div><dt>Detected languages</dt><dd>{countLabel(observedCounts(run, "detected_languages"))}</dd></div>}
                {run.progress && <div><dt>Progress</dt><dd>{run.progress.completed_samples}/{run.progress.total_samples ?? "?"}</dd></div>}
                <div><dt>Samples</dt><dd>{run.config.max_samples ?? "Full split"}</dd></div>
              </dl>
            </section>

            {(run.error || run.summary?.error) && <div className="error-panel"><AlertTriangle className="size-4" /><div><strong>Run failed</strong><p>{run.error || run.summary?.error}</p></div></div>}

            <section className="detail-section log-section">
              <div className="detail-section-title"><TerminalSquare className="size-4" />Live output</div>
              <pre aria-label="Evaluation logs">{logs || (ACTIVE_STATES.has(run.status) ? "Waiting for evaluator output…" : "No log output captured.")}</pre>
            </section>

            {run.artifacts.length > 0 && <section className="detail-section">
              <div className="detail-section-title"><Download className="size-4" />Artifacts</div>
              <div className="artifact-list">{run.artifacts.map((artifact) => <a key={artifact} href={`/api/runs/${run.id}/artifacts/${artifact}`}><span>{artifact}</span><Download className="size-3.5" /></a>)}</div>
            </section>}
          </div>
          {ACTIVE_STATES.has(run.status) && <div className="detail-footer"><Button variant="danger" onClick={cancel}><CircleStop className="size-4" />Cancel run</Button></div>}
        </>}
      </SheetContent>
    </Sheet>
  )
}

export default function App() {
  const [options, setOptions] = useState<Options | null>(null)
  const [runs, setRuns] = useState<Run[]>([])
  const [healthy, setHealthy] = useState(false)
  const [selected, setSelected] = useState<Run | null>(null)
  const [detailOpen, setDetailOpen] = useState(false)
  const [dark, setDark] = useState(document.documentElement.classList.contains("dark"))

  async function refresh() {
    try {
      const [nextOptions, nextRuns, health] = await Promise.all([api.options(), api.runs(), api.health()])
      setOptions(nextOptions)
      setRuns(nextRuns)
      setHealthy(health.status === "ok")
    } catch {
      setHealthy(false)
    }
  }

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 2500)
    return () => window.clearInterval(timer)
  }, [])

  function updateRun(updated: Run) {
    setRuns((current) => current.map((run) => run.id === updated.id ? updated : run))
    setSelected(updated)
  }

  function toggleTheme() {
    const next = !dark
    setDark(next)
    document.documentElement.classList.toggle("dark", next)
    localStorage.setItem("open-asr-theme", next ? "dark" : "light")
  }

  const activeRuns = runs.filter((run) => ACTIVE_STATES.has(run.status))
  const completedRuns = runs.filter((run) => !ACTIVE_STATES.has(run.status))

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand"><div className="brand-mark"><Activity className="size-4" /></div><div><strong>Open ASR</strong><span>Evaluation console</span></div></div>
        <div className="topbar-actions">
          <div className={`health-pill ${healthy ? "healthy" : "offline"}`}><span />{healthy ? "Server online" : "Disconnected"}</div>
          {options && <Popover>
            <PopoverTrigger asChild><Button variant="ghost" size="icon" aria-label="Credential status"><KeyRound className="size-4" /></Button></PopoverTrigger>
            <PopoverContent>
              <div className="popover-heading">Server credentials</div>
              <p className="popover-copy">Secret values never leave the server.</p>
              <div className="credential-list">{Object.entries(options.credentials).map(([name, configured]) => <div key={name}><span>{name}</span><Badge className={configured ? "credential-ok" : "credential-missing"}>{configured ? "Ready" : "Missing"}</Badge></div>)}</div>
            </PopoverContent>
          </Popover>}
          <Button variant="ghost" size="icon" onClick={toggleTheme} aria-label={dark ? "Use light theme" : "Use dark theme"}>{dark ? <Sun className="size-4" /> : <Moon className="size-4" />}</Button>
        </div>
      </header>

      <main className="workspace">
        <aside>{options ? <RunComposer options={options} activeCount={activeRuns.length} onCreated={(run) => { setRuns((current) => [run, ...current]); setSelected(run); setDetailOpen(true) }} /> : <div className="composer-card skeleton-card" />}</aside>
        <section className="runs-panel">
          <div className="runs-hero">
            <div><div className="eyebrow"><Server className="size-3.5" />Livekit server</div><h2>Evaluations</h2><p>Launch, watch, and inspect every ASR run from one place.</p></div>
            <div className="hero-stats"><div><Activity /><span>Active</span><strong>{activeRuns.length}</strong></div><div><Clock3 /><span>History</span><strong>{completedRuns.length}</strong></div></div>
          </div>

          {activeRuns.length > 0 && <div className="active-strip"><div className="pulse-orb" /><span>{activeRuns.length} evaluation{activeRuns.length === 1 ? " is" : "s are"} running in parallel</span><span className="active-models">{activeRuns.map((run) => `${configuredModelLabel(run.config.model_name)}: ${countLabel(observedCounts(run, "actual_models"))}`).join(" · ")}</span></div>}

          <div className="runs-heading"><div><h3>Run history</h3><p>Newest first · updates automatically</p></div><Button variant="outline" size="sm" onClick={refresh}><RefreshCw className="size-3.5" />Refresh</Button></div>
          <RunTable runs={runs} onSelect={(run) => { setSelected(run); setDetailOpen(true) }} />
        </section>
      </main>

      <RunDetail selected={selected} open={detailOpen} onOpenChange={setDetailOpen} onUpdated={updateRun} />
    </div>
  )
}
