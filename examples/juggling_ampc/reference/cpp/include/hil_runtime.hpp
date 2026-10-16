#pragma once
#include "controller.hpp"
#include <atomic>
#include <thread>
#include <array>
#include <string>
#include <vector>

namespace juggling {
static_assert(std::atomic<size_t>::is_always_lock_free && std::atomic<int>::is_always_lock_free);
template<class T,size_t Size> class SpscQueue {
 public:
  bool push(const T& value) noexcept {auto w=write_.load(std::memory_order_relaxed);auto next=(w+1)%Size;if(next==read_.load(std::memory_order_acquire))return false;data_[w]=value;write_.store(next,std::memory_order_release);return true;}
  bool pop(T& value) noexcept {auto r=read_.load(std::memory_order_relaxed);if(r==write_.load(std::memory_order_acquire))return false;value=data_[r];read_.store((r+1)%Size,std::memory_order_release);return true;}
  bool latest(T& value) noexcept {bool result=false;T next;while(pop(next)){value=next;result=true;}return result;}
 private:
  std::array<T,Size> data_{};
  alignas(64) std::atomic<size_t> write_{0};
  alignas(64) std::atomic<size_t> read_{0};
};
struct ArmFrame {double t=0;std::array<double,7> q{},dq{},torque{},reference_acceleration{};bool acceleration_input=false;};
struct StateFrame {double t=0;std::array<double,23> state{};std::array<double,7> torque{};};
struct CommandFrame {double source_time=0;std::array<double,84> trajectory{};}; // Four nodes: q,dq,tau (21 each).
struct HilStep {double source_time;StepInfo info;};

class VirtualBall {
 public:
  VirtualBall(const std::string& factors,const ArmFrame& initial);
  void advance(const ArmFrame& next);
  Joint prediction_torque(const ArmFrame& frame);
  const State& state() const {return state_;}
  int impacts() const {return impacts_;}
 private:
  RobotModel robot_;State state_;ArmFrame previous_;int impacts_=0;
  State free(const State& before,double start,double duration,const ArmFrame& next);
};

// One callback producer, one physics worker, one MPC worker. No MPC, CasADi,
// allocation, mutex, disk I/O or waiting occurs inside tick().
class HilRuntime {
 public:
  HilRuntime(const std::string& factors,const ArmFrame& initial,double max_age=.06);
  ~HilRuntime();
  void start();
  int tick(const ArmFrame& frame,std::array<double,7>& target,std::array<double,21>& reference,double& command_age) noexcept;
  void stop();
  void save(const std::string& directory);
  int fault() const {return fault_.load();}
 private:
  Controller controller_;VirtualBall ball_;ArmFrame initial_;
  SpscQueue<ArmFrame,8192> arms_;
  SpscQueue<StateFrame,32> states_;
  SpscQueue<CommandFrame,32> commands_;
  std::atomic<bool> running_{false};std::atomic<int> fault_{0};
  std::thread physics_,mpc_;
  std::vector<StateFrame> ball_log_;std::vector<HilStep> solve_log_;
  CommandFrame active_;double last_tick_=0,max_age_;
  int record_fault(int code) noexcept {
    int expected=0;
    fault_.compare_exchange_strong(expected,code);
    return expected==0?code:expected;
  }
  void physics_loop();
  void mpc_loop();
};
}
