const API_URL = "http://127.0.0.1:5000/predict";

const form = document.getElementById("predictForm");
const messageBox = document.getElementById("messageBox");
const statusValue = document.getElementById("statusValue");
const scoreValue = document.getElementById("scoreValue");

function showMessage(type, message) {
  messageBox.className = `alert alert-${type}`;
  messageBox.textContent = message;
  messageBox.classList.remove("d-none");
}

function hideMessage() {
  messageBox.classList.add("d-none");
  messageBox.textContent = "";
}

function setResult(status, score) {
  statusValue.textContent = status ?? "—";
  scoreValue.textContent =
    typeof score === "number" ? score.toFixed(6) : (score ?? "—");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideMessage();

  const formData = new FormData(form);
  const payload = {};

  for (const [key, value] of formData.entries()) {
    if (key === "star_spectype") {
      payload[key] = value.trim();
    } else {
      const num = Number(value);
      if (Number.isNaN(num)) {
        showMessage("danger", `Invalid value for ${key}. Please enter a valid number.`);
        return;
      }
      payload[key] = num;
    }
  }

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const data = await response.json();

    if (!response.ok) {
      showMessage("danger", data.message || "Backend error occurred.");
      setResult("—", "—");
      return;
    }

    const statusText =
      data.prediction === 1
        ? "Potentially Habitable"
        : "Non-Habitable";

    showMessage("success", "Prediction completed successfully.");
    setResult(statusText, data.habitability_probability);
  } catch (error) {
    showMessage("danger", "Could not connect to backend. Make sure Flask is running.");
    setResult("—", "—");
  }
});