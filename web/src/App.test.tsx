import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { vi } from "vitest"
import App from "./App"

const options = {
  providers: [{ prefix: "deepgram", label: "Deepgram", models: ["nova-3"], configured: true }],
  preprocessors: ["none", "arctan", "rnnoise"],
  vad_positions: ["none", "pre", "post"],
  credentials: { DEEPGRAM_API_KEY: true, ARCTAN_SDK_KEY: true, HF_TOKEN: true },
  defaults: { model_name: "deepgram/nova-3", dataset: "default", split: "test", max_workers: 4 },
}

const createdRun = {
  id: "abc123",
  status: "queued",
  created_at: "2026-07-10T00:00:00Z",
  started_at: null,
  finished_at: null,
  output_dir: "/tmp/abc123",
  config: {
    dataset_path: "bettercallaaryan/nc_agent_clips_openasr",
    dataset: "default",
    split: "test",
    model_name: "deepgram/nova-3",
    max_samples: null,
    max_workers: 4,
    use_url: false,
    streaming: false,
    prompt: null,
    audio_preprocessor: "arctan",
    vad_position: "post",
    arctan_chunk_ms: 10,
    audio_preprocessor_batch_size: 1,
  },
  summary: null,
  error: null,
  artifacts: [],
}

class FakeEventSource {
  addEventListener = vi.fn()
  close = vi.fn()
}

function jsonResponse(body: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }))
}

describe("Open ASR console", () => {
  beforeEach(() => {
    vi.stubGlobal("EventSource", FakeEventSource)
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const path = String(input)
      if (path === "/api/options") return jsonResponse(options)
      if (path === "/api/runs" && init?.method === "POST") return jsonResponse(createdRun, 201)
      if (path === "/api/runs") return jsonResponse([])
      if (path === "/api/health") return jsonResponse({ status: "ok", active_runs: 0 })
      if (path === "/api/datasets/inspect") return jsonResponse({ dataset_path: createdRun.config.dataset_path, configs: [{ name: "default", splits: ["test"], features: ["audio", "text"] }] })
      return jsonResponse({ detail: "not found" }, 404)
    }))
  })

  afterEach(() => vi.unstubAllGlobals())

  it("renders the configured pipeline and starts a run", async () => {
    const user = userEvent.setup()
    render(<App />)

    expect(await screen.findByText("Configure a run")).toBeInTheDocument()
    expect(screen.getByText("Audio → Processor → VAD → ASR")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Run evaluation" }))

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith(
        "/api/runs",
        expect.objectContaining({ method: "POST" }),
      )
    })
    expect((await screen.findAllByText("deepgram/nova-3")).length).toBeGreaterThan(0)
  })

  it("inspects a dataset and shows its schema", async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole("button", { name: /Inspect/i }))

    expect(await screen.findByText(/audio, text/)).toBeInTheDocument()
  })
})
