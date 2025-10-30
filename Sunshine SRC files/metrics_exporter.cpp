// src/metrics_exporter.cpp
#include "metrics_exporter.h"

#include <atomic>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <cstdlib>

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

namespace metrics {

static mutex g_mtx;
static atomic<bool> g_opened{false};
static ofstream g_f_video;
static ofstream g_f_input;
static steady_clock::time_point g_start_steady{steady_clock::now()};

static inline uint64_t now_epoch_ms() {
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}
static inline uint64_t now_monotonic_ms() {
  return duration_cast<milliseconds>(steady_clock::now() - g_start_steady).count();
}

static fs::path default_dir() {
#ifdef _WIN32
  const char* progdata = std::getenv("PROGRAMDATA");
  fs::path base = progdata ? fs::path(progdata) : fs::path("C:\\ProgramData");
  return base / "Sunshine" / "metrics";
#else
  const char* home = std::getenv("HOME");
  fs::path base = home ? fs::path(home) : fs::path("/tmp");
  return base / ".local" / "share" / "sunshine" / "metrics";
#endif
}

static std::string stamp_now_yyyymmdd_hhmmss() {
  auto now = system_clock::now();
  std::time_t tt = system_clock::to_time_t(now);
  std::tm tm{};
#ifdef _WIN32
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::ostringstream oss;
  oss << std::put_time(&tm, "%Y%m%d-%H%M%S");
  return oss.str();
}

static void ensure_open_nolock() {
  if (g_opened.load()) return;

  fs::path dir;
  if (const char* d = std::getenv("SUNSHINE_METRICS_DIR"); d && *d) dir = d;
  else dir = default_dir();
  fs::create_directories(dir);

  const std::string stamp = stamp_now_yyyymmdd_hhmmss();

  if (!g_f_video.is_open()) {
    fs::path p = dir / (stamp + ".video.csv");
    g_f_video.open(p.string(), std::ios::out | std::ios::trunc);
    g_f_video << "ts_epoch_ms,ts_monotonic_ms,frame_nr,port,bytes\n";
  }
  if (!g_f_input.is_open()) {
    fs::path p = dir / (stamp + ".input.csv");
    g_f_input.open(p.string(), std::ios::out | std::ios::trunc);
    // rtt_ms/var e loss_pct podem vir vazios dependendo do ENet
    g_f_input << "ts_epoch_ms,ts_monotonic_ms,rtt_ms,rtt_var_ms,loss_pct,channel_id,bytes\n";
  }

  g_opened.store(true);
}

void ensure_open() {
  if (g_opened.load()) return;
  std::lock_guard<std::mutex> lk(g_mtx);
  if (!g_opened.load()) ensure_open_nolock();
}

/* Compat: mantemos as APIs mas não bloqueamos logging. */
bool enabled() { return true; }
void enable(bool /*on*/) { ensure_open(); }

void log_video_frame(uint64_t frame_nr, uint32_t udp_port, size_t bytes) {
  ensure_open();
  const uint64_t t1 = now_epoch_ms();
  const uint64_t t2 = now_monotonic_ms();
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_f_video.is_open()) {
    g_f_video << t1 << ',' << t2 << ','
              << frame_nr << ',' << udp_port << ',' << bytes << '\n';
  }
}

void log_input_enet_sample(uint32_t rtt_ms, uint32_t rtt_var_ms, int loss_pct,
                           uint8_t channel_id, size_t bytes_rx) {
  ensure_open();
  const uint64_t t1 = now_epoch_ms();
  const uint64_t t2 = now_monotonic_ms();
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_f_input.is_open()) {
    g_f_input << t1 << ',' << t2 << ','
              << rtt_ms << ',' << rtt_var_ms << ',';
    if (loss_pct < 0) g_f_input << ""; else g_f_input << loss_pct;
    g_f_input << ',' << static_cast<unsigned>(channel_id) << ',' << bytes_rx << '\n';
  }
}

void flush_and_close() {
  std::lock_guard<std::mutex> lk(g_mtx);
  if (g_f_video.is_open()) g_f_video.flush();
  if (g_f_input.is_open()) g_f_input.flush();
}

} // namespace metrics
