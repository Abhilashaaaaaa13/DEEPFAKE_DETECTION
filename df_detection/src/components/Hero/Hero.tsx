import { useState } from "react";
import { sendToBackend } from "../../ml/sendToBackend";
import { MdArrowOutward } from "react-icons/md";

// --- START: NEW COMPONENTS & TYPES FOR RESULT VIEW ---

// New type to simulate richer data from the backend needed for the detailed view
type ResultType = {
  prediction: "REAL" | "UNCERTAIN" | "FAKE"; // More constrained type for scene logic
  confidence: number; // For the AUTH/RISK/FAKE percentage
  // Placeholder fields to represent the detailed signals from the image
  signals: {
    top: { label: string; value: string; isGood: boolean }[];
    warnings: { label: string; value: string; isWarning: boolean }[];
    anomalies: { label: string; value: string; isCritical: boolean }[];
  };
};

/**
 * Renders a single row for a signal, warning, or anomaly.
 */
const SignalRow = ({ label, value, color }: { label: string; value: string; color: string }) => (
  <div className="flex justify-between items-center py-1 border-b border-gray-700 last:border-b-0">
    <span className="text-gray-300 flex items-center">
      <span className={`h-2 w-2 rounded-full mr-2 ${color}`} />
      {label}
    </span>
    <span className="font-semibold text-white">{value}</span>
  </div>
);

/**
 * Main component to display the detailed result based on the image design.
 */
const ResultView = ({ result, onReset }: { result: ResultType; onReset: () => void }) => {
  const { prediction, confidence, signals } = result;

  // --- Determine Scene and Styling ---
  let sceneConfig: {
    title: string;
    authLabel: string;
    color: string;
    bgColor: string;
    signalsList: { title: string; list: { label: string; value: string; color: string }[] }[];
    bottomMessage: string;
  };

  if (prediction === "REAL") {
    sceneConfig = {
      title: "REAL CONTENT",
      authLabel: "AUTH",
      color: "text-emerald-400",
      bgColor: "bg-emerald-600",
      signalsList: [
        {
          title: "TOP SIGNALS",
          list: signals.top.map(s => ({ ...s, color: s.isGood ? "bg-emerald-400" : "bg-red-500" })),
        },
      ],
      bottomMessage: "No manipulation detected. Safe to view and share.",
    };
  } else if (prediction === "UNCERTAIN") {
    sceneConfig = {
      title: "UNCERTAIN",
      authLabel: "RISK",
      color: "text-yellow-400",
      bgColor: "bg-yellow-600",
      signalsList: [
        {
          title: "WARNING SIGNS",
          list: signals.warnings.map(w => ({ ...w, color: w.isWarning ? "bg-yellow-400" : "bg-emerald-400" })),
        },
      ],
      bottomMessage: "Inconclusive due to quality. Proceed with caution.",
    };
  } else { // FAKE
    sceneConfig = {
      title: "FAKE DETECTED",
      authLabel: "FAKE",
      color: "text-red-400",
      bgColor: "bg-red-600",
      signalsList: [
        {
          title: "CRITICAL ANOMALIES",
          list: signals.anomalies.map(a => ({ ...a, color: a.isCritical ? "bg-red-500" : "bg-emerald-400" })),
        },
      ],
      bottomMessage: "Strong evidence of AI synthesis found.",
    };
  }

  return (
    <div className="w-full h-full flex flex-col items-center justify-start gap-4 p-4 overflow-y-auto">
      {/* --- Main Result Panel --- */}
      <div className={`w-full max-w-sm rounded-xl p-4 shadow-2xl bg-gray-800 border-2 ${sceneConfig.color.replace('text', 'border')}`}>

        {/* 🔴 Title Header */}
        <div className={`flex items-center justify-center p-2 rounded-lg mb-4 ${sceneConfig.bgColor.replace('-600', '-700')} bg-opacity-30`}>
          <span className={`text-sm font-bold ${sceneConfig.color} mr-2`}>
            {sceneConfig.title}
          </span>
        </div>

        {/* 📊 Percentage Circle - MODIFIED */}
        <div className="flex flex-col items-center mb-6">
          <div className="relative h-28 w-28"> 
            {/* The outer ring/background circle */}
            <div className={`absolute top-0 left-0 h-full w-full rounded-full ${sceneConfig.bgColor} bg-opacity-30`} />
            
            {/* The inner circle with the percentage value */}
            <div className={`absolute top-1 left-1 h-[104px] w-[104px] bg-gray-800 rounded-full flex flex-col items-center justify-center border-4 ${sceneConfig.color.replace('text', 'border')}`}> 
              <span className={`text-4xl font-bold ${sceneConfig.color}`}>
                {confidence}%
              </span>
              <span className="text-xs text-gray-400">{sceneConfig.authLabel}</span>
            </div>
          </div>
        </div>
        {/* END MODIFIED PERCENTAGE CIRCLE */}

        {/* 📋 Signals/Warnings/Anomalies List */}
        <div className="space-y-4">
          {sceneConfig.signalsList.map((section, index) => (
            <div key={index} className="rounded-lg p-3 bg-gray-700/50">
              <h3 className="text-xs font-bold text-gray-400 mb-2 border-b border-gray-600 pb-1">
                {section.title}
              </h3>
              <div className="divide-y divide-gray-700">
                {section.list.map((signal, sIndex) => (
                  <SignalRow
                    key={sIndex}
                    label={signal.label}
                    value={signal.value}
                    color={signal.color}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
        
        {/* ℹ️ Bottom Message */}
        <div className={`mt-4 p-3 rounded-lg ${sceneConfig.bgColor} `}>
            <p className="text-sm text-center font-medium text-white">{sceneConfig.bottomMessage}</p>
        </div>


      </div>

      {/* 🔁 Reset Button */}
      <button
        onClick={onReset}
        className="mt-4 px-6 py-2 rounded-2xl bg-gray-600 text-white hover:bg-gray-500 transition"
      >
        Analyze Another Video
      </button>
    </div>
  );
};

// --- END: NEW COMPONENTS & TYPES FOR RESULT VIEW ---


const Hero = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoBlob, setVideoBlob] = useState<Blob | null>(null);

  const [recording, setRecording] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  // Initial state for result is null
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
      // Removed: const simpleRes = await sendToBackend(videoBlob); // TS6133: 'simpleRes' is declared but its value is never read.
      
      // In a real application, we would use:
      // const actualRes = await sendToBackend(videoBlob);
      // ... and then transform actualRes into the detailed ResultType.
      
      // Since we are mocking the result for demonstration:
      await sendToBackend(videoBlob); // Still call the backend if needed for side effects/progress, but ignore its simple return value

      // --- MOCKING THE FULL RESULT STRUCTURE ---
      const mockResult: ResultType = {
        // Randomly assign one of the three scenarios and a confidence value
        prediction: ["REAL", "UNCERTAIN", "FAKE"][Math.floor(Math.random() * 3)] as ResultType['prediction'],
        confidence: Math.floor(Math.random() * (100 - 45 + 1)) + 45, // Confidence between 45 and 100
        signals: {
          top: [
            { label: "Skin Texture", value: "Natural", isGood: true },
            { label: "Lip Sync", value: "Matched", isGood: true },
          ],
          warnings: [
            { label: "Lighting", value: "Mixed", isWarning: true },
            { label: "Compression", value: "High", isWarning: true },
          ],
          anomalies: [
            { label: "Blink Rate", value: "Static", isCritical: true },
            { label: "Audio Spec", value: "TTS Gen", isCritical: true },
          ],
        },
      };

      // Ensure confidence aligns with the mock prediction for display (as shown in image)
      if (mockResult.prediction === "REAL") mockResult.confidence = 98;
      if (mockResult.prediction === "UNCERTAIN") mockResult.confidence = 45;
      if (mockResult.prediction === "FAKE") mockResult.confidence = 99;


      setResult(mockResult); 
      // --- END MOCK ---

    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setAnalyzing(false);
    }
  };

  const handleReset = () => {
    setResult(null);
    setVideoUrl(null);
    setVideoBlob(null);
  };

  // 🎯 RESULT VIEW 
  if (result) {
    // Pass the full result object and the reset function to the new ResultView
    return <ResultView result={result} onReset={handleReset} />;
  }


  // 🎬 NORMAL FLOW UI
  return (
    <div className="w-full  mt-6 flex flex-col items-center gap-4 animate-slide-up">

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
          className={`px-6 py-2 flex font-semibold rounded-[10px] text-[16px] text-white transition ${
            recording
              ? "bg-gray-500"
              : "bg-blue-900 hover:scale-105"
          }`}
        >
          {recording ? "Recording..." : "Capture Video"}
          <div className="rounded-full h-[20px] w-[20px] ml-[10px] mt-[4px] pl-[1px] pt-[1px] justify-center items-center bg-white text-blue-900">
            <MdArrowOutward />
          </div>
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