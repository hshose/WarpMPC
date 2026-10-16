#pragma once
#include "model_config.hpp"
#include <array>
#ifndef JUGGLING_BALL_GRAVITY_X
#define JUGGLING_BALL_GRAVITY_X 0.
#define JUGGLING_BALL_GRAVITY_Y 0.
#define JUGGLING_BALL_GRAVITY_Z -9.81
#endif
namespace juggling {
inline const std::array<double,3> ball_gravity={JUGGLING_BALL_GRAVITY_X,JUGGLING_BALL_GRAVITY_Y,JUGGLING_BALL_GRAVITY_Z};
}
