const state = {
  imageDataUrl: "",
  imageFile: null,
  config: null,
  questions: [],
};

const els = {
  modeStatus: document.querySelector("#modeStatus"),
  dropZone: document.querySelector("#dropZone"),
  imageInput: document.querySelector("#imageInput"),
  previewWrap: document.querySelector("#previewWrap"),
  previewImage: document.querySelector("#previewImage"),
  fileName: document.querySelector("#fileName"),
  fileMeta: document.querySelector("#fileMeta"),
  form: document.querySelector("#analysisForm"),
  lightSelect: document.querySelector("#lightSelect"),
  categorySelect: document.querySelector("#categorySelect"),
  questionSelect: document.querySelector("#questionSelect"),
  questionText: document.querySelector("#questionText"),
  runButton: document.querySelector("#runButton"),
  modelGrid: document.querySelector("#modelGrid"),
  promptBox: document.querySelector("#promptBox"),
};

const formatBytes = (bytes) => {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB"];
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`;
};

async function getJson(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function renderModels(models) {
  els.modelGrid.innerHTML = models
    .map(
      (model) => `
        <article class="model-card" style="--accent:${model.accent}">
          <div class="model-card__top">
            <div>
              <h3>${model.label}</h3>
              <p>${model.model_id}</p>
            </div>
            <span>${model.benchmark_accuracy.toFixed(2)}%</span>
          </div>
          <output>${model.answer || "Waiting"}</output>
          <div class="meter" aria-label="Confidence">
            <span style="width:${model.confidence || 0}%"></span>
          </div>
          <div class="model-meta">
            <span>${model.confidence || 0}% confidence</span>
            <span>${model.latency_ms || "-"} ms</span>
          </div>
        </article>
      `
    )
    .join("");
}

async function loadConfig() {
  state.config = await getJson("/api/config");
  els.modeStatus.textContent = `${state.config.mode} mode`;
  els.categorySelect.innerHTML = state.config.categories
    .map((category) => `<option value="${category}">${category}</option>`)
    .join("");
  renderModels(state.config.models.map((model) => ({ ...model, benchmark_accuracy: 0 })));
  await loadQuestions();
}

async function loadQuestions() {
  const category = els.categorySelect.value;
  const data = await getJson(`/api/questions?category=${encodeURIComponent(category)}`);
  state.questions = data.questions;
  els.questionSelect.innerHTML = state.questions
    .map((row, index) => `<option value="${index}">${row.question}</option>`)
    .join("");
  syncQuestionText();
}

function syncQuestionText() {
  const row = state.questions[Number(els.questionSelect.value)];
  els.questionText.value = row?.question || "";
}

function setImage(file) {
  if (!file || !file.type.startsWith("image/")) return;
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    state.imageFile = file;
    state.imageDataUrl = String(reader.result);
    els.previewImage.src = state.imageDataUrl;
    els.fileName.textContent = file.name;
    els.fileMeta.textContent = formatBytes(file.size);
    els.previewWrap.hidden = false;
  });
  reader.readAsDataURL(file);
}

async function runAnalysis(event) {
  event.preventDefault();
  if (!state.imageDataUrl) {
    els.promptBox.textContent = "Drop an image first.";
    return;
  }

  els.runButton.disabled = true;
  els.runButton.querySelector("span").textContent = "Running...";
  els.promptBox.textContent = "Building prompt and sending image to all four model slots...";

  try {
    const data = await getJson("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_data_url: state.imageDataUrl,
        image_name: state.imageFile?.name,
        light: els.lightSelect.value,
        category: els.categorySelect.value,
        question: els.questionText.value,
      }),
    });
    renderModels(data.results);
    els.promptBox.textContent = data.prompt;
  } catch (error) {
    els.promptBox.textContent = error.message;
  } finally {
    els.runButton.disabled = false;
    els.runButton.querySelector("span").textContent = "Run all models";
  }
}

els.imageInput.addEventListener("change", (event) => setImage(event.target.files[0]));
els.dropZone.addEventListener("dragover", (event) => {
  event.preventDefault();
  els.dropZone.classList.add("is-dragging");
});
els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("is-dragging"));
els.dropZone.addEventListener("drop", (event) => {
  event.preventDefault();
  els.dropZone.classList.remove("is-dragging");
  setImage(event.dataTransfer.files[0]);
});
els.categorySelect.addEventListener("change", loadQuestions);
els.questionSelect.addEventListener("change", syncQuestionText);
els.form.addEventListener("submit", runAnalysis);

loadConfig().catch((error) => {
  els.modeStatus.textContent = "offline";
  els.promptBox.textContent = error.message;
});
