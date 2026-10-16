// Offline device double for the *unmodified* all-C++ hardware supervisor.
// No libfranka, Vicon SDK, sockets or physical device access is linked here.
#include "experiment_math.hpp"
#include "native_transport.hpp"
#include "session_control.hpp"
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <thread>
#include <iostream>
namespace {
using namespace juggling;
namespace ex=juggling::experiment;
namespace nt=juggling::native;
namespace g=juggling::generated;
namespace fs=std::filesystem;
using Clock=std::chrono::steady_clock;
double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
volatile std::sig_atomic_t stopped=0;
void stop(int){stopped=1;}
struct World {
 std::uint64_t magic=0x4a55474d4f434b31ULL;
 std::atomic<std::uint64_t> revision{0};
 std::array<std::atomic<double>,7> q{},dq{},ddq{};
 std::atomic<std::uint32_t> robot_started{0},plans_received{0},moves_received{0},stop_seen{0};
};
class WorldRegion {
 int fd_=-1;World* world_=nullptr;
 public:
 WorldRegion(const fs::path& path,bool create){
  fd_=::open(path.c_str(),O_RDWR|O_CLOEXEC|O_NOFOLLOW|(create?O_CREAT|O_EXCL:0),0600);
  if(fd_<0)throw std::runtime_error("Cannot open mock world");
  if(create&&ftruncate(fd_,sizeof(World)))throw std::runtime_error("Cannot size mock world");
  struct stat status{};if(fstat(fd_,&status)||status.st_size!=sizeof(World))throw std::runtime_error("Invalid mock world file");
  void* p=mmap(nullptr,sizeof(World),PROT_READ|PROT_WRITE,MAP_SHARED,fd_,0);
  if(p==MAP_FAILED)throw std::runtime_error("Cannot map mock world");
  world_=create?new(p)World{}:static_cast<World*>(p);
  if(world_->magic!=0x4a55474d4f434b31ULL)throw std::runtime_error("Mock world protocol mismatch");
 }
 ~WorldRegion(){if(world_)munmap(world_,sizeof(World));if(fd_>=0)::close(fd_);}
 World& get(){return *world_;}
};
void publish(World& w,const Joint& q,const Joint& dq,const Joint& ddq){
 w.revision.fetch_add(1,std::memory_order_acq_rel);
 for(int j=0;j<7;++j){w.q[j].store(q[j],std::memory_order_relaxed);w.dq[j].store(dq[j],std::memory_order_relaxed);w.ddq[j].store(ddq[j],std::memory_order_relaxed);}
 w.revision.fetch_add(1,std::memory_order_release);
}
Joint snapshot(World& w){Joint q;for(;;){const auto before=w.revision.load(std::memory_order_acquire);if(before%2)continue;for(int j=0;j<7;++j)q[j]=w.q[j].load(std::memory_order_relaxed);if(before==w.revision.load(std::memory_order_acquire))return q;}}
struct Setup {
 ex::pt::ptree config,fixed,metadata;
 fs::path factors,output,world_path;
 RobotModel model;
 ex::Pose F_T_P,P_T_CAD,V_T_B;
 Eigen::Matrix3d normal_rotation=Eigen::Matrix3d::Identity();
 explicit Setup(const fs::path& path):config(ex::read_json(path)),factors(config.get<std::string>("factors")),output(config.get<std::string>("output")),world_path(config.get<std::string>("world")),model(factors.string()){
  fixed=ex::read_json(config.get<std::string>("calibration"));metadata=ex::read_json(factors/"metadata.json");F_T_P=ex::pose(fixed.get_child("F_T_P"));P_T_CAD=ex::pose(fixed.get_child("P_T_CAD_face"));
  V_T_B=ex::pose(fixed.get_child("calibration_V_T_B"));
  if(auto value=metadata.get_child_optional("spec.geometry.paddle_normal_rotvec_deg")){Eigen::Vector3d v=ex::vector<3>(*value)*(M_PI/180);if(v.norm()>1e-12)normal_rotation=Eigen::AngleAxisd(v.norm(),v.normalized()).toRotationMatrix();}
 }
 ex::Pose flange(const Joint& q){auto fk=model.contact_fk(q);ex::Pose face=ex::Pose::Identity();face.topLeftCorner<3,3>()=fk[4]*normal_rotation.transpose();face.topRightCorner<3,1>()=fk[0];return face*(F_T_P*P_T_CAD).inverse();}
};
void array(std::ostream& stream,const std::array<double,7>& values){stream<<'[';for(int j=0;j<7;++j){if(j)stream<<',';stream<<values[j];}stream<<']';}
void write_packet(const nt::ViconPacket& packet){const char* p=reinterpret_cast<const char*>(&packet);std::size_t count=sizeof(packet);while(count){const auto sent=::write(1,p,count);if(sent<0&&errno==EINTR){if(stopped)return;continue;}if(sent<=0)throw std::runtime_error("Mock Vicon pipe closed");p+=sent;count-=sent;}}
void name(std::array<char,96>& to,const std::string& from){if(from.size()>=to.size())throw std::runtime_error("Oversized mock name");std::copy(from.begin(),from.end(),to.begin());}
int vicon(Setup& setup){WorldRegion region(setup.world_path,false);auto& world=region.get();std::uint64_t frame=0;double released=0;auto next=Clock::now();
 const Eigen::Vector3d anchor=ex::map(g::origin)+Eigen::Vector3d(0.,0.,.8),gravity(g::gravity_x,g::gravity_y,g::gravity_z);
 while(!stopped){
  nt::ViconPacket p;p.receive_time=now();p.frame=++frame;p.latency_valid=1;p.latency_s=.005;p.frame_rate_hz=100.;p.paddle_valid=1;p.unlabeled_available=1;
  name(p.subject,"hsh_pingpong_paddle");name(p.segment,setup.fixed.get<std::string>("paddle_segment"));
  const ex::Pose paddle=setup.V_T_B*setup.flange(snapshot(world))*setup.F_T_P;
  Eigen::Matrix<double,4,4,Eigen::RowMajor> row_major=paddle;std::copy(row_major.data(),row_major.data()+16,p.paddle_T.begin());
  const Eigen::Vector3d position=paddle.topRightCorner<3,1>();std::copy(position.data(),position.data()+3,p.paddle_position_m.begin());
  const Eigen::Quaterniond rotation(paddle.topLeftCorner<3,3>());std::copy(rotation.coeffs().data(),rotation.coeffs().data()+4,p.paddle_quaternion_xyzw.begin());
  if(!released&&fs::exists(setup.output/"ready.json"))released=now()+.3;
  const double capture=p.receive_time-.003;const double t=released?std::max(0.,capture-released):0.;
  if(t<.25){
   const double drag=g::drag;const double a=drag>1e-8?-std::expm1(-drag*t)/drag:t;const double b=drag>1e-8?(t-a)/drag:.5*t*t;
   const Eigen::Vector3d ball=setup.V_T_B.topLeftCorner<3,3>()*(anchor+b*gravity)+setup.V_T_B.topRightCorner<3,1>();
   p.unlabeled_count=1;p.unlabeled[0].id=111;std::copy(ball.data(),ball.data()+3,p.unlabeled[0].position_m.begin());
  }
  write_packet(p);next+=std::chrono::milliseconds(10);if(next<Clock::now()-std::chrono::milliseconds(20))next=Clock::now();std::this_thread::sleep_until(next);
 }
 return 0;
}
int robot(Setup& setup,int argc,char** argv){std::string ipc,output,host;double delivery=.035;
 for(int i=1;i<argc;++i){std::string a=argv[i];if((a=="--ipc"||a=="--output"||a=="--robot"||a=="--max-plan-delivery"||a=="--seconds")&&i+1<argc){const std::string value=argv[++i];if(a=="--ipc")ipc=value;else if(a=="--output")output=value;else if(a=="--robot")host=value;else if(a=="--max-plan-delivery")delivery=std::stod(value);}}
 if(host!="mock"||ipc.empty()||output.empty())throw std::runtime_error("Mock robot requires --robot mock and native IPC");
 if(!fs::create_directory(output))throw std::runtime_error("Mock robot output must be new");
 auto channel=nt::SharedRegion::open(ipc);auto& shared=channel.get();WorldRegion region(setup.world_path,false);auto& world=region.get();
 Joint q=snapshot(world),dq=Joint::Zero(),ddq=Joint::Zero();std::array<double,7> home;std::copy(q.data(),q.data()+7,home.begin());SessionControl control(home);control.delivery_limit(delivery);
 std::ofstream log(fs::path(output)/"robot.jsonl");std::string failure;int fault=0;const double began=now();double previous=began;auto next=Clock::now();world.robot_started=1;shared.producer_state=1;
 try{while(true){const double t=now(),dt=t-previous;previous=t;bool ending=stopped||shared.stop.load(std::memory_order_acquire);
   if(session_heartbeat_expired(t,shared.heartbeat.load(std::memory_order_acquire),ending))throw std::runtime_error("Mock robot heartbeat expired");
   SessionCommand command;while(shared.commands.pop(command)){
    if(command.kind==SessionCommand::Plan)++world.plans_received;else if(command.kind==SessionCommand::Move)++world.moves_received;
    std::array<double,7> measured;std::copy(q.data(),q.data()+7,measured.begin());control.accept(command,measured,t);
   }
   if(ending){world.stop_seen=1;dq.setZero();ddq.setZero();}else{const auto reference=control.reference(t);for(int j=0;j<7;++j){q[j]=reference[j];dq[j]=reference[7+j];ddq[j]=reference[14+j];}}
   publish(world,q,dq,ddq);nt::NativePacket p;p.receive_time=t;p.elapsed=t-began;p.dt=dt;p.move_id=control.id();p.moving=control.moving(t);p.mpc=control.mpc();p.stopping=ending;p.reference_valid=!ending;p.success_rate=1.;
   std::copy(q.data(),q.data()+7,p.q.begin());p.q_d=p.q;std::copy(dq.data(),dq.data()+7,p.dq.begin());p.dq_d=p.dq;std::copy(ddq.data(),ddq.data()+7,p.ddq_d.begin());
   const Joint tau=setup.model.rnea(q,dq,ddq);std::copy(tau.data(),tau.data()+7,p.tau_J.begin());
   ex::Pose flange=setup.flange(q),identity=ex::Pose::Identity();std::copy(flange.data(),flange.data()+16,p.O_T_EE.begin());std::copy(identity.data(),identity.data()+16,p.F_T_EE.begin());
   if(!shared.states.push(p))throw std::runtime_error("Mock robot feedback overflow");
   log<<std::setprecision(17)<<"{\"receive_time\":"<<t<<",\"mock_device\":true,\"q\":";array(log,p.q);log<<",\"dq\":";array(log,p.dq);log<<"}\n";
   if(ending)break;next+=std::chrono::milliseconds(2);if(next<Clock::now()-std::chrono::milliseconds(5))next=Clock::now();std::this_thread::sleep_until(next);
  }}catch(const std::exception& e){failure=e.what();fault=90;shared.fault=fault;std::cerr<<failure<<'\n';}
 shared.producer_state=fault?3:2;log.close();
 ex::write(fs::path(output)/"session.json","{\"mock_device\":true,\"control_completed\":"+std::string(fault?"false":"true")+",\"fault\":"+std::to_string(fault)+",\"plans_received\":"+std::to_string(world.plans_received.load())+",\"moves_received\":"+std::to_string(world.moves_received.load())+",\"stop_seen\":"+std::to_string(world.stop_seen.load())+",\"error\":"+json_text(failure)+"}");return fault?1:0;
}
}
int main(int argc,char** argv){try{
 if(argc==2&&std::string(argv[1])=="--describe"){std::cout<<"{\"mock_devices\":true,\"robot_connected\":false,\"network_capability\":false,\"protocol\":2,\"model_sha256\":\""<<JUGGLING_MODEL_SHA256<<"\"}\n";return 0;}
 if(!std::getenv("JUGGLING_MOCK_DEVICES")||std::string(std::getenv("JUGGLING_MOCK_DEVICES"))!="1")throw std::runtime_error("Offline mock requires JUGGLING_MOCK_DEVICES=1");
 const char* config=std::getenv("JUGGLING_MOCK_CONFIG");if(!config)throw std::runtime_error("Missing JUGGLING_MOCK_CONFIG");Setup setup(config);
 std::signal(SIGINT,stop);std::signal(SIGTERM,stop);std::signal(SIGPIPE,SIG_IGN);
 if(argc==2&&std::string(argv[1])=="--init"){WorldRegion region(setup.world_path,true);publish(region.get(),ex::map(g::home),Joint::Zero(),Joint::Zero());return 0;}
 if(argc>1&&std::string(argv[1])=="mock")return vicon(setup);
 if(argc>1&&std::string(argv[1])=="--execute")return robot(setup,argc,argv);
 throw std::runtime_error("Mock devices accept --init, mock Vicon args, or --execute --robot mock");
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
