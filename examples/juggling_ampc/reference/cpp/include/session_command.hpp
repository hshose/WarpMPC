#pragma once
#include <array>

namespace juggling {
// Shared native/text protocol. Four nodes of q, dq, torque and acceleration.
struct SessionCommand {
  enum Kind { Move, Plan, Stop } kind=Stop;
  double source=0;
  int id=0;
  std::array<double,7> goal{};
  static constexpr int stride=28;
  std::array<double,4*stride> trajectory{};
};
}
