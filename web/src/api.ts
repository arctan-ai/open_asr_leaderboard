export type Provider = {
  prefix: string
  label: string
  models: string[]
  language_options: Record<string, {
    batch: LanguageOption[]
    streaming: LanguageOption[]
  }>
  configured: boolean
}
export type LanguageOption = { code: string; label: string }

export type DatasetSource = {
  id: "huggingface" | "local"
  label: string
  kind: "huggingface" | "local"
  description: string
}

export type DatasetOption = {
  id: string
  label: string
  dataset_source: "huggingface" | "local"
  dataset_path: string
  dataset: string
  splits: string[]
  features: string[]
  valid: boolean | null
  error: string | null
  validation_status?: "unchecked" | "valid" | "invalid"
}

export type DatasetCatalog = { source_id: string; datasets: DatasetOption[] }
export type AuthUser = { id: string; email: string; name: string }


export type Options = {
  providers: Provider[]
  preprocessors: string[]
  vad_positions: string[]
  dataset_sources: DatasetSource[]
  credentials: Record<string, boolean>
  defaults: { dataset_source: "huggingface" | "local"; model_name: string; dataset: string; split: string; max_workers: number }
}

export type RunConfig = {
  dataset_path: string
  dataset_source: "huggingface" | "local"
  dataset: string
  split: string
  model_name: string
  language: string
  max_samples: number | null
  max_workers: number
  use_url: boolean
  streaming: boolean
  prompt: string | null
  audio_preprocessor: string
  vad_position: string
  arctan_chunk_ms: number
  audio_preprocessor_batch_size: number
}

export type RunSummary = {
  wer_percent?: number
  rtfx?: number
  num_samples?: number
  error?: string
  actual_models?: Record<string, number>
  detected_languages?: Record<string, number>
}

export type RunProgress = {
  completed_samples: number
  total_samples: number | null
  actual_models: Record<string, number>
  detected_languages: Record<string, number>
}

export type Run = {
  id: string
  status: "queued" | "running" | "cancelling" | "completed" | "failed" | "cancelled" | "interrupted"
  created_at: string
  started_at: string | null
  finished_at: string | null
  output_dir: string
  config: RunConfig
  summary: RunSummary | null
  progress: RunProgress | null
  error: string | null
  artifacts: string[]
}

export class ApiError extends Error {
  code?: string

  constructor(message: string, code?: string) {
    super(message)
    this.name = "ApiError"
    this.code = code
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  })
  if (response.status === 401) {
    globalThis.location.assign("/auth/google/login")
    throw new ApiError("Authentication required")
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    const detail = body.detail
    const message = typeof detail === "string" ? detail : detail?.message
    const code = typeof detail === "object" ? detail?.code : undefined
    throw new ApiError(message || "Request failed", code)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  me: () => request<AuthUser>("/api/auth/me"),
  logout: () => request<void>("/api/auth/logout", { method: "POST" }),
  options: () => request<Options>("/api/options"),
  health: () => request<{ status: string; active_runs: number }>("/api/health"),
  runs: () => request<Run[]>("/api/runs"),
  run: (id: string) => request<Run>(`/api/runs/${id}`),
  createRun: (config: RunConfig) => request<Run>("/api/runs", { method: "POST", body: JSON.stringify(config) }),
  cancelRun: (id: string) => request<Run>(`/api/runs/${id}/cancel`, { method: "POST" }),
  retryRun: (id: string) => request<Run>(`/api/runs/${id}/retry`, { method: "POST" }),
  inspectDataset: (dataset_path: string) => request<{ dataset_path: string; configs: { name: string; splits: string[]; features: string[]; schema?: Record<string, unknown> }[] }>("/api/datasets/inspect", { method: "POST", body: JSON.stringify({ dataset_path }) }),
  datasets: (source_id: string) => request<DatasetCatalog>(`/api/datasets?source_id=${encodeURIComponent(source_id)}`),
}
