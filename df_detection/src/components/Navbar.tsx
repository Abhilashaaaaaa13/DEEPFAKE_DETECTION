const Navbar = () => {
  return (
    <div className="w-full flex items-center justify-between">

      {/* Left: Avatar + Text */}
      <div className="flex flex-row gap-3 items-center">
        <div className="h-[40px] w-[40px] rounded-full bg-red-400 text-amber-50 flex items-center justify-center font-bold">
          D
        </div>

        <div className="text-white leading-tight">
          <h3 className="text-[12px] opacity-80">Good Morning.</h3>
          <h1 className="text-[18px] font-semibold">Dhruv Barthwal</h1>
        </div>
      </div>

      {/* ❌ Close Button */}
      <button
        onClick={() => window.close()}
        className="text-white text-xl font-bold
        hover:text-red-400 transition
        rounded-full w-8 h-8 flex items-center justify-center"
        aria-label="Close Extension"
      >
        ✕
      </button>

    </div>
  );
};

export default Navbar;
