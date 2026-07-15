import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { vi } from "vitest"
import App from "./App"

const options = {
  providers: [
    {
      prefix: "deepgram", label: "Deepgram", models: ["nova-3"], configured: true,
      language_options: { "nova-3": { batch: [{ code: "multi", label: "Multilingual / code-switching" }, { code: "en", label: "English" }, { code: "es", label: "Spanish" }], streaming: [{ code: "multi", label: "Multilingual / code-switching" }, { code: "en", label: "English" }] } },
    },
    {
      prefix: "sarvam", label: "Sarvam", models: ["saaras:v3"], configured: true,
      language_options: { "saaras:v3": { batch: [{ code: "unknown", label: "Automatic detection" }, { code: "en-IN", label: "English (India)" }], streaming: [{ code: "unknown", label: "Automatic detection" }, { code: "en-IN", label: "English (India)" }] } },
    },
  ],
  preprocessors: ["none", "arctan", "rnnoise"],
  vad_positions: ["none", "pre", "post"],
  credentials: { DEEPGRAM_API_KEY: true, SARVAM_API_KEY: true, ARCTAN_SDK_KEY: true, HF_TOKEN: true },
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
    const createCall = vi.mocked(fetch).mock.calls.find(([path, init]) => path === "/api/runs" && init?.method === "POST")
    expect(JSON.parse(String(createCall?.[1]?.body))).toEqual(expect.objectContaining({ language: "en" }))
  })

  it("inspects a dataset and shows its schema", async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(await screen.findByRole("button", { name: /Inspect/i }))

    expect(await screen.findByText(/audio, text/)).toBeInTheDocument()
  })

  it("submits Sarvam automatic language detection as unknown", async () => {
    const user = userEvent.setup()
    render(<App />)

    await screen.findByText("Configure a run")
    await user.click(screen.getAllByRole("combobox")[0])
    await user.click(await screen.findByRole("option", { name: "Sarvam · saaras:v3" }))
    await user.click(screen.getAllByRole("combobox")[1])
    await user.click(await screen.findByRole("option", { name: "Automatic detection (unknown)" }))
    await user.click(screen.getByRole("button", { name: "Run evaluation" }))

    await waitFor(() => {
      const createCall = vi.mocked(fetch).mock.calls.find(([path, init]) => path === "/api/runs" && init?.method === "POST")
      expect(JSON.parse(String(createCall?.[1]?.body))).toEqual(expect.objectContaining({ model_name: "sarvam/saaras:v3", language: "unknown" }))
    })
  })

  it("updates language options when the model changes", async () => {
    const user = userEvent.setup()
    render(<App />)

    await screen.findByText("Configure a run")
    await user.click(screen.getAllByRole("combobox")[1])
    expect(await screen.findByRole("option", { name: "Spanish (es)" })).toBeInTheDocument()
    await user.click(screen.getByRole("option", { name: "English (en)" }))

    await user.click(screen.getAllByRole("combobox")[0])
    await user.click(await screen.findByRole("option", { name: "Sarvam · saaras:v3" }))

    await waitFor(() => {
      expect(screen.getAllByRole("combobox")[1]).toHaveTextContent("Automatic detection")
    })
    await user.click(screen.getAllByRole("combobox")[1])
    expect(screen.queryByRole("option", { name: "Spanish (es)" })).not.toBeInTheDocument()
    expect(await screen.findByRole("option", { name: "English (India) (en-IN)" })).toBeInTheDocument()
  })

  it("resets a language that is unsupported in streaming mode", async () => {
    const user = userEvent.setup()
    render(<App />)

    await screen.findByText("Configure a run")
    await user.click(screen.getAllByRole("combobox")[1])
    await user.click(await screen.findByRole("option", { name: "Spanish (es)" }))
    await user.click(screen.getByRole("switch", { name: "Streaming endpoint" }))

    await waitFor(() => {
      expect(screen.getAllByRole("combobox")[1]).toHaveTextContent("Multilingual / code-switching")
    })
  })
})
