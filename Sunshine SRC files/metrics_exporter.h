#pragma once
#include <cstdint>
#include <string_view>
#include <cstddef>

namespace metrics {

// Pode ativar/desativar por env var SUNSHINE_METRICS=1
bool enabled();
void enable(bool on);

// Inicialização preguiçosa: abre arquivos em diretório dado por
//   SUNSHINE_METRICS_DIR (ou padrão do sistema) e escreve cabeçalho CSV.
void ensure_open();

// Loga um frame de vídeo no momento em que for chamado.
void log_video_frame(uint64_t frame_nr, uint32_t stream_port /*ou 0*/, size_t bytes_hint = 0);

// Loga um pacote de input (comandos) com tipo (packetType do NV_INPUT_HEADER) e tamanho.
//void log_input_packet(uint16_t packet_type, size_t payload_bytes);


// NOVO: “amostrar” estatísticas do ENet peer ao receber qualquer pacote de input
// rtt_ms:       ENetPeer->roundTripTime (ms)
// rtt_var_ms:   ENetPeer->roundTripTimeVariance (ms) [opcional]
// loss_pct:     ENetPeer->packetLoss (0..100) se disponível; senão passe -1
// channel_id:   event.channelID do ENet
// bytes_rx:     event.packet->dataLength
void log_input_enet_sample(uint32_t rtt_ms, uint32_t rtt_var_ms, int loss_pct,
                           uint8_t channel_id, size_t bytes_rx);


// Opcional: fechar/flush explícito ao encerrar sessão
void flush_and_close();
}
