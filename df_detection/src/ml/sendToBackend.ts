type BackendResult = {
  prediction: string;
  confidence: number;
};

export async function sendToBackend(
  videoBlob: Blob
): Promise<BackendResult> {
  const formData = new FormData();

  // MUST match FastAPI parameter name
  formData.append("file", videoBlob, "recorded_video.webm");

  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    body: formData
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(errorText || "Backend request failed");
  }

  const result: BackendResult = await response.json();

  console.log("🧠 Deepfake Result:", result);

  return result;
}
