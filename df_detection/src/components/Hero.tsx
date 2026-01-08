import { useState } from "react";
import { sendToBackend } from "../ml/sendToBackend";

const Hero = () => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);

  const handleAddVideo = async () => {
    try {
      setVideoUrl(null);
      setRecording(true);

      // 1️⃣ User selects tab/screen
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

      // 2️⃣ Stop after 5 seconds
      setTimeout(() => {
        mediaRecorder.stop();
        stream.getTracks().forEach(track => track.stop());
      }, 5000);

      mediaRecorder.onstop = async () => {
        setRecording(false);

        const videoBlob = new Blob(chunks, { type: "video/webm" });
        const url = URL.createObjectURL(videoBlob);

        // 3️⃣ Show video in popup
        setVideoUrl(url);

        // 4️⃣ Send to backend (optional: delay or button)
        await sendToBackend(videoBlob);
      };

    } catch (error) {
      setRecording(false);
      console.error("Recording failed or cancelled", error);
    }
  };

  return (
    <div className="w-full h-full mt-6 flex flex-col items-center gap-4">
      
      {/* Headings */}
      <div className="flex flex-col items-center">
        <h1 className="font-bold text-white text-2xl">
          Seeing Is Believing?
        </h1>
        <h2 className="text-white text-xl">
          Detect Now.
        </h2>
      </div>

      {/* Button */}
      <button
        onClick={handleAddVideo}
        disabled={recording}
        className={`px-6 py-2 rounded-2xl text-white ${
          recording ? "bg-gray-500" : "bg-red-400"
        }`}
      >
        {recording ? "Recording..." : "Add Video"}
      </button>

      {/* Recording indicator */}
      {recording && (
        <p className="text-red-400 font-semibold">
          ● Recording for 5 seconds
        </p>
      )}

      {/* Recorded video preview */}
      {videoUrl && (
        <div className="w-full mt-2">
          <p className="text-white text-sm mb-1">Recorded Clip:</p>
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
