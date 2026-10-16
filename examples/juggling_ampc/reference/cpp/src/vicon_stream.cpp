// Read-only Vicon SDK stream. No robot connection or motion capability.
#include "ViconDataStreamSDK_CPP/DataStreamClient.h"
#include "json_text.hpp"
#include "native_transport.hpp"
#include <chrono>
#include <cmath>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace ViconDataStreamSDK::CPP;
using Clock=std::chrono::steady_clock;
static volatile std::sig_atomic_t stopped=0;
static void stop(int){stopped=1;}
static double now(){return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();}
template<size_t N> bool finite(const double (&a)[N]){for(double x:a)if(!std::isfinite(x))return false;return true;}
template<size_t N> void array(const double (&a)[N],double scale=1){std::cout<<'[';for(size_t i=0;i<N;++i){if(i)std::cout<<',';std::cout<<a[i]*scale;}std::cout<<']';}
template<std::size_t N> void fixed_name(std::array<char,N>& destination,const std::string& value){
 if(value.size()>=N)throw std::runtime_error("Vicon name exceeds native protocol capacity");
 std::copy(value.begin(),value.end(),destination.begin());
}
static juggling::native::ViconPacket packet(Client& c,const std::string& subject,double received,unsigned frame){
 juggling::native::ViconPacket out;out.receive_time=received;out.frame=frame;fixed_name(out.subject,subject);
 auto latency=c.GetLatencyTotal();out.latency_valid=latency.Result==Result::Success&&std::isfinite(latency.Total);
 if(out.latency_valid)out.latency_s=latency.Total;
 auto rate=c.GetFrameRate();if(rate.Result==Result::Success&&std::isfinite(rate.FrameRateHz)&&rate.FrameRateHz>0)out.frame_rate_hz=rate.FrameRateHz;
 auto root=c.GetSubjectRootSegmentName(subject);
 if(root.Result==Result::Success){
  fixed_name(out.segment,std::string(root.SegmentName));
  auto p=c.GetSegmentGlobalTranslation(subject,root.SegmentName);auto q=c.GetSegmentGlobalRotationQuaternion(subject,root.SegmentName);
  if(p.Result==Result::Success&&q.Result==Result::Success&&!p.Occluded&&!q.Occluded&&finite(p.Translation)&&finite(q.Rotation)){
   double norm=0;for(double x:q.Rotation)norm+=x*x;
   if(norm>1e-12){
    norm=std::sqrt(norm);for(unsigned k=0;k<4;++k)out.paddle_quaternion_xyzw[k]=q.Rotation[k]/norm;
    for(unsigned k=0;k<3;++k)out.paddle_position_m[k]=p.Translation[k]*.001;
    const auto& a=out.paddle_quaternion_xyzw;const double x=a[0],y=a[1],z=a[2],w=a[3];
    out.paddle_T={1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),out.paddle_position_m[0],
      2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),out.paddle_position_m[1],
      2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),out.paddle_position_m[2],0,0,0,1};
    out.paddle_valid=1;
   }
  }
 }
 auto count=c.GetSubjectCount();
 if(count.Result!=Result::Success)throw std::runtime_error("Vicon subject list unavailable");
 for(unsigned s=0;s<count.SubjectCount;++s){
  auto name=c.GetSubjectName(s);if(name.Result!=Result::Success)throw std::runtime_error("Vicon subject name unavailable");
  auto n=c.GetMarkerCount(name.SubjectName);if(n.Result!=Result::Success)throw std::runtime_error("Vicon labeled marker count unavailable");
  for(unsigned m=0;m<n.MarkerCount;++m){
   if(out.labeled_count==out.labeled.size())throw std::runtime_error("Vicon labeled marker native packet capacity exceeded (256)");
   auto marker=c.GetMarkerName(name.SubjectName,m);if(marker.Result!=Result::Success)throw std::runtime_error("Vicon marker name unavailable");
   auto p=c.GetMarkerGlobalTranslation(name.SubjectName,marker.MarkerName);auto& destination=out.labeled[out.labeled_count++];
   fixed_name(destination.subject,std::string(name.SubjectName));fixed_name(destination.name,std::string(marker.MarkerName));
   destination.valid=p.Result==Result::Success&&!p.Occluded&&finite(p.Translation);
   if(destination.valid)for(unsigned k=0;k<3;++k)destination.position_m[k]=p.Translation[k]*.001;
  }
 }
 auto raw=c.GetUnlabeledMarkerCount();out.unlabeled_available=raw.Result==Result::Success;
 if(out.unlabeled_available){
  if(raw.MarkerCount>out.unlabeled.size())throw std::runtime_error("Vicon unlabeled marker native packet capacity exceeded (512)");
  for(unsigned m=0;m<raw.MarkerCount;++m){auto p=c.GetUnlabeledMarkerGlobalTranslation(m);
   if(p.Result!=Result::Success||!finite(p.Translation))continue;
   auto& destination=out.unlabeled[out.unlabeled_count++];destination.id=p.MarkerID;
   for(unsigned k=0;k<3;++k)destination.position_m[k]=p.Translation[k]*.001;
  }
 }
 return out;
}
static void write_packet(const juggling::native::ViconPacket& packet){
 const auto* bytes=reinterpret_cast<const char*>(&packet);std::size_t remaining=sizeof(packet);
 while(remaining){const auto count=::write(STDOUT_FILENO,bytes,remaining);
  if(count<0&&errno==EINTR){if(stopped)return;continue;}
  if(count<=0)throw std::runtime_error("Vicon binary output closed");
  bytes+=count;remaining-=static_cast<std::size_t>(count);
 }
}
int main(int argc,char** argv){
 if(argc==2&&std::string(argv[1])=="--describe"){
  std::cout<<"{\"protocol\":"<<juggling::native::protocol_version<<",\"binary_packet_bytes\":"<<sizeof(juggling::native::ViconPacket)<<",\"robot_connected\":false}\n";return 0;
 }
 bool binary=false;std::vector<std::string> positional;
 for(int i=1;i<argc;++i){if(std::string(argv[i])=="--binary")binary=true;else positional.emplace_back(argv[i]);}
 if(positional.empty()||positional.size()>3){std::cout<<"Read-only: juggling_vicon HOST[:801] [SECONDS=0] [PADDLE_SUBJECT=hsh_pingpong_paddle] [--binary]\n";return argc==1?0:2;}
 try{
  double seconds=positional.size()>1?std::stod(positional[1]):0;
  if(!std::isfinite(seconds)||seconds<0)throw std::runtime_error("Invalid duration");
  std::string subject=positional.size()>2?positional[2]:"hsh_pingpong_paddle";
  std::signal(SIGINT,stop);std::signal(SIGTERM,stop);std::signal(SIGPIPE,SIG_IGN);Client c;double start=now();
  while(!stopped&&!c.IsConnected().Connected){c.Connect(positional[0]);if(now()-start>8)throw std::runtime_error("Vicon connection timeout");std::this_thread::sleep_for(std::chrono::milliseconds(20));}
  if(stopped)return 0;
  auto check=[](Result::Enum result){if(result!=Result::Success)throw std::runtime_error("Vicon configuration failed");};
  check(c.EnableSegmentData().Result);check(c.EnableMarkerData().Result);check(c.EnableUnlabeledMarkerData().Result);
  bool segments=c.IsSegmentDataEnabled().Enabled,labeled=c.IsMarkerDataEnabled().Enabled,raw=c.IsUnlabeledMarkerDataEnabled().Enabled;
  std::cerr<<std::boolalpha<<"{\"segment_data_enabled\":"<<segments<<",\"labeled_marker_data_enabled\":"<<labeled<<",\"unlabeled_marker_data_enabled\":"<<raw<<"}\n";
  if(!segments||!labeled||!raw)throw std::runtime_error("Required Vicon data subscription is not enabled");
  check(c.SetStreamMode(StreamMode::ServerPush).Result);
  check(c.SetAxisMapping(Direction::Forward,Direction::Left,Direction::Up).Result);
  std::cout<<std::setprecision(17);start=now();double last=start;unsigned previous=0;bool have=false;
  while(!stopped&&(seconds==0||now()-start<seconds)){
   if(c.GetFrame().Result!=Result::Success){if(now()-last>2)throw std::runtime_error("Vicon frame timeout");std::this_thread::sleep_for(std::chrono::milliseconds(1));continue;}
   double received=now();auto number=c.GetFrameNumber();if(number.Result!=Result::Success)continue;
   if(have&&number.FrameNumber==previous){if(now()-last>2)throw std::runtime_error("Frozen Vicon frame counter");continue;}
   have=true;previous=number.FrameNumber;last=received;
   if(binary){write_packet(packet(c,subject,received,number.FrameNumber));continue;}
   std::cout<<"{\"receive_time\":"<<received<<",\"frame\":"<<number.FrameNumber<<",\"latency_s\":";
   auto latency=c.GetLatencyTotal();if(latency.Result==Result::Success&&std::isfinite(latency.Total))std::cout<<latency.Total;else std::cout<<"null";
   std::cout<<",\"frame_rate_hz\":";auto rate=c.GetFrameRate();if(rate.Result==Result::Success)std::cout<<rate.FrameRateHz;else std::cout<<"null";
   std::cout<<",\"paddle\":";auto root=c.GetSubjectRootSegmentName(subject);bool valid=false;
   if(root.Result==Result::Success){auto p=c.GetSegmentGlobalTranslation(subject,root.SegmentName);auto q=c.GetSegmentGlobalRotationQuaternion(subject,root.SegmentName);
    valid=p.Result==Result::Success&&q.Result==Result::Success&&!p.Occluded&&!q.Occluded&&finite(p.Translation)&&finite(q.Rotation);
    if(valid){std::cout<<"{\"subject\":"<<juggling::json_text(subject)<<",\"segment\":"<<juggling::json_text(root.SegmentName)<<",\"position_m\":";array(p.Translation,.001);std::cout<<",\"quaternion_xyzw\":";array(q.Rotation);std::cout<<'}';}}
   if(!valid)std::cout<<"null";
   std::cout<<",\"subjects\":[";auto count=c.GetSubjectCount().SubjectCount;
   for(unsigned s=0;s<count;++s){if(s)std::cout<<',';std::cout<<juggling::json_text(c.GetSubjectName(s).SubjectName);}std::cout<<"],\"markers\":[";bool first=true;
   for(unsigned s=0;s<count;++s){auto name=c.GetSubjectName(s).SubjectName;auto n=c.GetMarkerCount(name).MarkerCount;
    for(unsigned m=0;m<n;++m){auto marker=c.GetMarkerName(name,m).MarkerName;auto p=c.GetMarkerGlobalTranslation(name,marker);
     // Retain names even while occluded. Never replace occlusion with [0,0,0].
     if(!first)std::cout<<',';first=false;std::cout<<"{\"subject\":"<<juggling::json_text(name)<<",\"name\":"<<juggling::json_text(marker)<<",\"position_m\":";
     if(p.Result==Result::Success&&!p.Occluded&&finite(p.Translation))array(p.Translation,.001);else std::cout<<"null";std::cout<<'}';}}
   // Reconstructed 3D points carry position and MarkerID, not physical size.
   // Per-camera centroid radii are image measurements, not ball diameters.
   std::cout<<"],\"unlabeled_marker_diameter_available\":false,\"unlabeled_markers\":";
   auto unlabeled=c.GetUnlabeledMarkerCount();
   if(unlabeled.Result!=Result::Success)std::cout<<"null";
   else{std::cout<<'[';bool raw_first=true;for(unsigned m=0;m<unlabeled.MarkerCount;++m){auto p=c.GetUnlabeledMarkerGlobalTranslation(m);
    if(p.Result!=Result::Success||!finite(p.Translation))continue;
    if(!raw_first)std::cout<<',';raw_first=false;std::cout<<"{\"id\":"<<p.MarkerID<<",\"position_m\":";array(p.Translation,.001);std::cout<<'}';}
    std::cout<<']';}
   std::cout<<"}\n"<<std::flush;if(!std::cout)throw std::runtime_error("Vicon output closed");
  }
  c.Disconnect();return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
