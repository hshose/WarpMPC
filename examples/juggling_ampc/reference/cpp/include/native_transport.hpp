#pragma once
// Typed Linux IPC for the hardware supervisor. No serialization, allocation,
// mutex, filesystem operation or syscall is needed by the robot callback.
#include "session_command.hpp"
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <type_traits>
#include <unistd.h>

namespace juggling::native {
inline constexpr std::uint64_t robot_magic=0x4a554747524f4231ULL;
inline constexpr std::uint64_t vicon_magic=0x4a55474756494331ULL;
inline constexpr std::uint32_t protocol_version=1;

template<class T,std::size_t N> class SpscQueue {
  static_assert(N>1&&std::is_trivially_copyable<T>::value);
  alignas(64) std::atomic<std::uint64_t> read_{0};
  alignas(64) std::atomic<std::uint64_t> write_{0};
  alignas(64) std::array<T,N> values_{};
 public:
  bool push(const T& value) noexcept {
    const auto w=write_.load(std::memory_order_relaxed);
    if(w-read_.load(std::memory_order_acquire)>=N)return false;
    values_[w%N]=value;write_.store(w+1,std::memory_order_release);return true;
  }
  bool pop(T& value) noexcept {
    const auto r=read_.load(std::memory_order_relaxed);
    if(r==write_.load(std::memory_order_acquire))return false;
    value=values_[r%N];read_.store(r+1,std::memory_order_release);return true;
  }
};

struct NativePacket {
  double receive_time=0,elapsed=0,dt=0;
  std::int32_t move_id=0,fault=0;
  std::uint32_t moving=0,mpc=0,stopping=0,reference_valid=0;
  double mpc_plan_age_s=0;
  std::array<double,7> q{},dq{},q_d{},dq_d{},ddq_d{},tau_J{};
  // Franka's column-major transforms, exactly as reported by the robot.
  std::array<double,16> O_T_EE{},F_T_EE{};
  std::array<double,7> velocity_command{},guard_filtered_torque{},guard_filtered_torque_rate{};
  std::array<double,21> reference_q_dq_ddq{};
  double robot_time=0,success_rate=0;
};

struct SharedRobot {
  std::uint64_t magic=robot_magic;
  std::uint32_t version=protocol_version;
  std::uint32_t bytes=sizeof(SharedRobot);
  alignas(64) std::atomic<double> heartbeat{0};
  std::atomic<std::uint32_t> stop{0};
  // 0 startup; 1 callback active; 2 clean exit; 3 failure. One writer: robot.
  std::atomic<std::uint32_t> producer_state{0};
  std::atomic<std::uint32_t> fault{0};
  SpscQueue<SessionCommand,32> commands;
  SpscQueue<NativePacket,2048> states;
};
static_assert(std::atomic<double>::is_always_lock_free);
static_assert(std::atomic<std::uint32_t>::is_always_lock_free);
static_assert(std::atomic<std::uint64_t>::is_always_lock_free);
static_assert(std::is_trivially_copyable<NativePacket>::value);

class SharedRegion {
  int fd_=-1;
  SharedRobot* data_=nullptr;
  explicit SharedRegion(int fd,SharedRobot* data):fd_(fd),data_(data){}
  static std::runtime_error error(const char* operation){
    return std::runtime_error(std::string(operation)+": "+std::strerror(errno));
  }
 public:
  SharedRegion()=default;
  SharedRegion(const SharedRegion&)=delete;
  SharedRegion& operator=(const SharedRegion&)=delete;
  SharedRegion(SharedRegion&& other)noexcept:fd_(other.fd_),data_(other.data_){other.fd_=-1;other.data_=nullptr;}
  SharedRegion& operator=(SharedRegion&& other)noexcept{
    if(this!=&other){close();fd_=other.fd_;data_=other.data_;other.fd_=-1;other.data_=nullptr;}return *this;
  }
  ~SharedRegion(){close();}
  void close()noexcept{if(data_)::munmap(data_,sizeof(SharedRobot));if(fd_>=0)::close(fd_);data_=nullptr;fd_=-1;}
  static SharedRegion create(const std::string& path){
    const int fd=::open(path.c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
    if(fd<0)throw error("Create robot shared memory");
    if(::ftruncate(fd,sizeof(SharedRobot))!=0){const auto e=error("Size robot shared memory");::close(fd);::unlink(path.c_str());throw e;}
    void* memory=::mmap(nullptr,sizeof(SharedRobot),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(memory==MAP_FAILED){const auto e=error("Map robot shared memory");::close(fd);::unlink(path.c_str());throw e;}
    // Constructs atomics and prefaults the complete mapping before launching
    // the callback process. The file remains available for run provenance.
    auto* data=new(memory) SharedRobot{};
    return SharedRegion(fd,data);
  }
  static SharedRegion open(const std::string& path){
    const int fd=::open(path.c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW);
    if(fd<0)throw error("Open robot shared memory");
    struct stat status{};
    if(::fstat(fd,&status)!=0||!S_ISREG(status.st_mode)||status.st_size!=static_cast<off_t>(sizeof(SharedRobot))){::close(fd);throw std::runtime_error("Invalid robot shared-memory file size/type");}
    void* memory=::mmap(nullptr,sizeof(SharedRobot),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(memory==MAP_FAILED){const auto e=error("Map robot shared memory");::close(fd);throw e;}
    auto* data=static_cast<SharedRobot*>(memory);
    if(data->magic!=robot_magic||data->version!=protocol_version||data->bytes!=sizeof(SharedRobot)){
      ::munmap(memory,sizeof(SharedRobot));::close(fd);throw std::runtime_error("Robot shared-memory protocol mismatch");}
    // Touch the producer's mapping before entering robot control. MAP_SHARED
    // pages are already initialized by create(), but page tables are per process.
    volatile unsigned char touch=0;
    auto* bytes=static_cast<const unsigned char*>(memory);
    for(std::size_t k=0;k<sizeof(SharedRobot);k+=4096)touch^=bytes[k];
    (void)touch;
    return SharedRegion(fd,data);
  }
  SharedRobot& get(){if(!data_)throw std::runtime_error("Robot shared memory is not open");return *data_;}
  const SharedRobot& get()const{if(!data_)throw std::runtime_error("Robot shared memory is not open");return *data_;}
};

struct RawMarker {
  std::uint64_t id=0;
  std::array<double,3> position_m{};
};
struct LabeledMarker {
  std::array<char,96> subject{},name{};
  std::uint32_t valid=0;
  std::array<double,3> position_m{};
};
struct ViconPacket {
  std::uint64_t magic=vicon_magic;
  std::uint32_t version=protocol_version;
  std::uint32_t bytes=sizeof(ViconPacket);
  double receive_time=0;
  std::uint64_t frame=0;
  std::uint32_t latency_valid=0,paddle_valid=0,unlabeled_available=0;
  double latency_s=0,frame_rate_hz=0;
  std::array<char,96> subject{},segment{};
  // Row-major Vicon-world to paddle-subject transform (translation in metres).
  std::array<double,16> paddle_T{};
  std::array<double,3> paddle_position_m{};
  std::array<double,4> paddle_quaternion_xyzw{};
  std::uint32_t unlabeled_count=0,labeled_count=0;
  std::array<RawMarker,512> unlabeled{};
  std::array<LabeledMarker,256> labeled{};
};
static_assert(std::is_trivially_copyable<ViconPacket>::value);
inline void validate(const ViconPacket& packet){
  if(packet.magic!=vicon_magic||packet.version!=protocol_version||packet.bytes!=sizeof(ViconPacket)||
     packet.unlabeled_count>packet.unlabeled.size()||packet.labeled_count>packet.labeled.size())
    throw std::runtime_error("Vicon binary protocol mismatch");
}
}
