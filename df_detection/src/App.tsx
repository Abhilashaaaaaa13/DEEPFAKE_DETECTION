import Hero from "./components/Hero";
import Navbar from "./components/Navbar";
import "./index.css";

function App() {
  return (
    <div
      className="h-[340px] w-[350px]
      bg-gray-900                                 
      p-4                              
      flex flex-col gap-4"                 
    >
      <Navbar />
      <Hero />
    </div>
  );
}

export default App;
