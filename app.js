const pages = document.querySelectorAll(".page");
const navButtons = document.querySelectorAll("[data-target]");
const modeTabs = document.querySelectorAll(".tab");
const modePanels = document.querySelectorAll(".mode-panel");

const previewImage = document.getElementById("previewImage");
const emptyPreviewText = document.getElementById("emptyPreviewText");
const captionText = document.getElementById("captionText");
const confidenceText = document.getElementById("confidenceText");
const captionLoading = document.getElementById("captionLoading");

const fileInput = document.getElementById("fileInput");
const randomImageBtn = document.getElementById("randomImageBtn");
const urlInput = document.getElementById("urlInput");
const loadUrlBtn = document.getElementById("loadUrlBtn");
const startCameraBtn = document.getElementById("startCameraBtn");
const captureBtn = document.getElementById("captureBtn");
const webcamPreview = document.getElementById("webcamPreview");
const generateCaptionBtn = document.getElementById("generateCaptionBtn");

const searchInput = document.getElementById("searchInput");
const searchBtn = document.getElementById("searchBtn");
const resultsGrid = document.getElementById("resultsGrid");

let currentMode = "dataset";
let webcamStream = null;
let selectedImageSource = "";
let selectedDatasetImageName = "";
let selectedLocalFile = null;
let selectedWebcamDataUrl = "";

function navigateTo(pageId) {
  pages.forEach((page) => page.classList.remove("active"));
  const target = document.getElementById(pageId);
  if (target) target.classList.add("active");
}

navButtons.forEach((button) => {
  button.addEventListener("click", () => navigateTo(button.dataset.target));
});

function activateMode(mode) {
  currentMode = mode;
  modeTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === mode));
  modePanels.forEach((panel) => panel.classList.toggle("active", panel.id === `mode-${mode}`));
}

modeTabs.forEach((tab) => {
  tab.addEventListener("click", () => activateMode(tab.dataset.mode));
});

function setPreview(src) {
  if (!src) {
    previewImage.style.display = "none";
    previewImage.src = "";
    emptyPreviewText.style.display = "block";
    selectedImageSource = "";
    return;
  }
  selectedImageSource = src;
  previewImage.src = src;
  previewImage.style.display = "block";
  emptyPreviewText.style.display = "none";
}

async function loadRandomDatasetImage() {
  try {
    const response = await fetch("/api/random-image");
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Failed to get random image");
    selectedDatasetImageName = data.image_name;
    setPreview(data.image_url);
    captionText.textContent = "Dataset image ready. Click Generate Caption.";
    confidenceText.textContent = "";
  } catch (error) {
    captionText.textContent = error.message;
  }
}

randomImageBtn.addEventListener("click", loadRandomDatasetImage);

fileInput.addEventListener("change", (e) => {
  const file = e.target.files?.[0];
  if (!file) return;
  selectedLocalFile = file;
  const objectUrl = URL.createObjectURL(file);
  setPreview(objectUrl);
});

loadUrlBtn.addEventListener("click", () => {
  const value = urlInput.value.trim();
  if (!value) return;
  setPreview(value);
});

startCameraBtn.addEventListener("click", async () => {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    webcamPreview.srcObject = webcamStream;
  } catch (error) {
    captionText.textContent = "Camera access denied or unavailable.";
  }
});

captureBtn.addEventListener("click", () => {
  if (!webcamPreview.srcObject) return;
  const canvas = document.createElement("canvas");
  canvas.width = webcamPreview.videoWidth || 640;
  canvas.height = webcamPreview.videoHeight || 480;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(webcamPreview, 0, 0, canvas.width, canvas.height);
  selectedWebcamDataUrl = canvas.toDataURL("image/jpeg", 0.95);
  setPreview(selectedWebcamDataUrl);
});

function setCaptionLoading(isLoading) {
  if (isLoading) {
    captionLoading.classList.remove("hidden");
    captionText.textContent = "Generating caption...";
    confidenceText.textContent = "";
  } else {
    captionLoading.classList.add("hidden");
  }
}

async function generateCaptionRequest() {
  setCaptionLoading(true);
  try {
    let response;
    if (currentMode === "dataset") {
      if (!selectedDatasetImageName) await loadRandomDatasetImage();
      response = await fetch("/api/caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "dataset",
          image_name: selectedDatasetImageName
        })
      });
    } else if (currentMode === "local") {
      if (!selectedLocalFile) throw new Error("Please upload an image file first.");
      const form = new FormData();
      form.append("mode", "local");
      form.append("image", selectedLocalFile);
      response = await fetch("/api/caption", { method: "POST", body: form });
    } else if (currentMode === "webcam") {
      if (!selectedWebcamDataUrl) throw new Error("Capture an image from webcam first.");
      response = await fetch("/api/caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "webcam",
          data_url: selectedWebcamDataUrl
        })
      });
    } else {
      const imageUrl = urlInput.value.trim();
      if (!imageUrl) throw new Error("Paste a valid image URL first.");
      response = await fetch("/api/caption", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: "url",
          image_url: imageUrl
        })
      });
    }

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Caption generation failed");
    captionText.textContent = data.caption || "No caption generated.";
    confidenceText.textContent = "AI caption generated successfully";
  } catch (error) {
    captionText.textContent = error.message;
    confidenceText.textContent = "";
  } finally {
    setCaptionLoading(false);
  }
}

generateCaptionBtn.addEventListener("click", generateCaptionRequest);

function renderResults(items) {
  if (!items.length) {
    resultsGrid.innerHTML = '<p class="helper">No results found. Try a different query.</p>';
    return;
  }
  resultsGrid.innerHTML = items
    .map(
      (item) => `
      <article class="result-card fade-up">
        <img src="${item.src}" alt="${item.label}">
        <div class="result-meta">
          <p>${item.label}</p>
          <span>Similarity: ${item.score}</span>
        </div>
      </article>
    `
    )
    .join("");
}

async function searchImages() {
  const query = searchInput.value.trim();
  if (!query) {
    resultsGrid.innerHTML = '<p class="helper">Type a query and click search to see matching images.</p>';
    return;
  }

  resultsGrid.innerHTML = '<p class="helper">Searching...</p>';
  try {
    const response = await fetch(`/api/search?q=${encodeURIComponent(query)}&top_k=8`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Search failed");
    renderResults(data.results || []);
  } catch (error) {
    resultsGrid.innerHTML = `<p class="helper">${error.message}</p>`;
  }
}

searchBtn.addEventListener("click", searchImages);
searchInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") searchImages();
});

resultsGrid.innerHTML = '<p class="helper">Type a query and click search to see matching images.</p>';