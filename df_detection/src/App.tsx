import Hero from "./components/Hero/Hero";
import Navbar from "./components/Navbar";
import "./index.css";

function App() {
  return (
    <div
      className="relative min-h-[280px] w-[350px]
      bg-gray-900
      p-4
      flex flex-col gap-4"
    >
      {/* ✨ Glow Effect */}
      <div
        className="pointer-events-none absolute -top-24 left-1/2 -translate-x-1/2
        w-[420px] h-[220px]
        rounded-b-full
        bg-gradient-to-b from-pink-200/40 via-pink-100/20 to-transparent
        blur-3xl"
      />

      <Navbar />
      <Hero />
    </div>
  );
}

export default App;
