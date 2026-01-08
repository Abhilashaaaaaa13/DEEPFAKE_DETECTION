import { useState } from "react";
import { sendToBackend } from "../ml/sendToBackend";

type ResultType = {
  prediction: string;
  confidence: number;
};

const Hero = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);

  const [recording, setRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<ResultType | null>(null);

  // 🎥 Capture video
  const handleAddVideo = async () => {
    try {
      setVideoUrl(null);
      setVideoBlob(null);
      setResult(null);
      setRecording(true);

      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: false
      });

      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: "video/webm"
      });

      const chunks: BlobPart[] = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.start();

      setTimeout(() => {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }, 5000);

      mediaRecorder.onstop = () => {
        setRecording(false);

        const blob = new Blob(chunks, { type: "video/webm" });
        setVideoBlob(blob);
        setVideoUrl(URL.createObjectURL(blob));
      };

    } catch (error) {
      setRecording(false);
      console.error("Recording failed", error);
    }
  };

  // 🧠 Analyze video
  const handleAnalyzeVideo = async () => {
    if (!videoBlob) return;

    try {
      setAnalyzing(true);
      const res = await sendToBackend(videoBlob);
      setResult(res); // 👈 STORE RESULT
    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setAnalyzing(false);
    }
  };

  // 🎯 RESULT VIEW 
  if (result) {
    const isFake = result.prediction.includes("FAKE");

    return (
      <div className="w-full h-full flex flex-col items-center justify-center gap-4 animate-slide-up">

        <h1
          className={`text-3xl font-bold ${
            isFake ? "text-red-500" : "text-emerald-400"
          }`}
        >
          {result.prediction}
        </h1>

        <div className="w-full px-6">
          <div className="w-full bg-gray-700 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full ${
                isFake ? "bg-red-500" : "bg-emerald-500"
              }`}
              style={{ width: `${result.confidence}%` }}
            />
          </div>
          <p className="text-center text-white mt-2 font-semibold">
            Confidence: {result.confidence}%
          </p>
        </div>

        <button
          onClick={() => {
            setResult(null);
            setVideoUrl(null);
            setVideoBlob(null);
          }}
          className="mt-4 px-6 py-2 rounded-2xl bg-gray-600 text-white hover:bg-gray-500 transition"
        >
          Analyze Another Video
        </button>
      </div>
    );
  }

  // 🎬 NORMAL FLOW UI
  return (
    <div className="w-full h-full mt-6 flex flex-col items-center gap-4 animate-slide-up">

      {/* Headings */}
      <div className="flex flex-col items-center">
        <h1 className="font-bold text-white text-2xl">
          Seeing Is Believing?
        </h1>
        <h2 className="text-white text-xl">
          Detect Now.
        </h2>
      </div>

      {/* Buttons */}
      {!videoUrl ? (
        <button
          onClick={handleAddVideo}
          disabled={recording}
          className={`px-6 py-2 font-semibold rounded-2xl text-white transition ${
            recording
              ? "bg-gray-500"
              : "bg-red-500 hover:scale-105"
          }`}
        >
          {recording ? "Recording..." : "Capture Video"}
        </button>
      ) : (
        <button
          onClick={handleAnalyzeVideo}
          disabled={analyzing}
          className={`px-6 py-2 font-semibold rounded-2xl text-white transition ${
            analyzing
              ? "bg-gray-500"
              : "bg-emerald-500 hover:scale-105"
          }`}
        >
          {analyzing ? "Analyzing..." : "Analyze Video"}
        </button>
      )}

      {recording && (
        <p className="text-red-400 font-semibold animate-pulse">
          ● Recording for 5 seconds
        </p>
      )}

      {videoUrl && (
        <div className="w-full mt-2">
          <p className="text-white font-semibold mb-1">
            Recorded Clip:
          </p>
          <video
            src={videoUrl}
            controls
            className="w-full rounded-lg border border-gray-700"
          />
        </div>
      )}
    </div>
  );
};

export default Hero;
