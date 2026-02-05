function login() {
  window.location.href = "/home";
}

function goUpload(plant) {
  localStorage.setItem("plant", plant);
  window.location.href = "/upload";
}

function goHome() {
  window.location.href = "/home";
}

window.onload = function () {
  if (document.getElementById("plantTitle")) {
    document.getElementById("plantTitle").innerText =
      "Upload " + localStorage.getItem("plant") + " Leaf Image";
  }
};

async function predict() {
  const fileInput = document.getElementById("imageInput");
  const files = fileInput.files;

  if (files.length === 0) {
    alert("Please select an image first!");
    return;
  }

  const formData = new FormData();
  formData.append("image", files[0]);

  // Show loading state
  const btn = document.querySelector(".primary-btn");
  const originalText = btn.innerText;
  btn.innerText = "Analyzing...";
  btn.disabled = true;

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (response.ok) {
      document.getElementById("disease").innerText = data.prediction;
      document.getElementById("confidence").innerText = data.confidence;

      let treatments = "";
      if (data.info && data.info.treatment) {
        treatments = data.info.treatment.join(", ");
      } else {
        treatments = "No specific data.";
      }
      document.getElementById("treatment").innerText = treatments;

      document.getElementById("result").style.display = "block";
    } else {
      alert("Error: " + (data.error || "Unknown error occurred"));
    }

  } catch (error) {
    console.error("Error:", error);
    alert("An error occurred during prediction.");
  } finally {
    btn.innerText = originalText;
    btn.disabled = false;
  }
}

function openChatbot() {
  window.location.href = "/chatbot";
}

async function showWeather() {
  try {
    const response = await fetch('/api/get_weather');
    const data = await response.json();
    if (!data.error) {
      alert(`Weather: ${data.desc}, Temp: ${data.temp}°C, Humidity: ${data.humidity}%`);
    } else {
      alert("Weather data unavailable: " + data.error);
    }
  } catch (e) {
    alert("Could not fetch weather.");
  }
}

function showSatellite() {
  alert("Satellite NDVI: Moderate vegetation stress detected (Mock Data)");
}

function changeLanguage() {
  // Determine current lang and toggle
  // This requires backend support to know current or pass it in. 
  // For now, redirect to language select page
  window.location.href = "/language_select";
}
