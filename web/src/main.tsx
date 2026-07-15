import { StrictMode } from "react"
import { createRoot } from "react-dom/client"
import { Toaster } from "sonner"
import App from "./App"
import "./index.css"

const savedTheme = localStorage.getItem("open-asr-theme") || "dark"
document.documentElement.classList.toggle("dark", savedTheme === "dark")

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
    <Toaster richColors position="bottom-right" closeButton />
  </StrictMode>,
)
