# SubjectSieve

SubjectSieve is a Double-DQN controller for cloud gaming that jointly selects bitrate and FPS. Optimizes a QoE-driven reward, enforces bitrate-first under low QoE, and uses action masking/hysteresis, replay, and target networks for stable, sample-efficient adaptation.

# How to use

An small Jupyter Notebook file with all the necessary steps are included.

#Modifing Sunshine:

Build (Windows, MSYS2 UCRT64)

1 -> Install & open UCRT64 shell
2 -> Install [MSYS2], then launch MSYS2 UCRT64 (not MINGW64).
3 -> Update: pacman -Syu
4 -> install dependencies: pacman -S --needed mingw-w64-ucrt-x86_64-toolchain mingw-w64-ucrt-x86_64-cmake mingw-w64-ucrt-x86_64-ninja mingw-w64-ucrt-x86_64-pkgconf mingw-w64-ucrt-x86_64-git mingw-w64-ucrt-x86_64-openssl mingw-w64-ucrt-x86_64-opus mingw-w64-ucrt-x86_64-libvpx mingw-w64-ucrt-x86_64-miniupnpc mingw-w64-ucrt-x86_64-libnatpmp mingw-w64-ucrt-x86_64-nlohmann-json mingw-w64-ucrt-x86_64-json
5->Get Sunshine: git clone --recursive https://github.com/LizardByte/Sunshine.git
6 -> cd Sunshine
7 -> copy and replace files in this git
8 -> cmake -G Ninja -B build -S . -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DOPENSSL_USE_STATIC_LIBS=ON
9 ->  ninja -C build
10 -> copy encoder_runtime.json to the folder of sunshine.exe

Any problems are probably due to missing dependencies (Sunshine needs a lot of them)
