export async function sendToBackend(videoBlob: Blob) {
  const formData = new FormData();
  formData.append("video", videoBlob, "captured_clip.webm");

  try {
    const response = await fetch("http://localhost:8000/detect", {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      throw new Error("Backend error");
    }

    const result = await response.json();

    console.log("🧠 Deepfake Result:", result);

    chrome.notifications?.create({
      type: "basic",
      iconUrl: "icon.png",
      title: "Deepfake Detection",
      message: `Probability: ${result.probability}`
    });

  } catch (error) {
    console.error("❌ Backend request failed", error);
  }
}
