#include "native_transport.hpp"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sys/wait.h>
#include <thread>

using namespace juggling;
using namespace juggling::native;
static void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
int main(){
 const std::string path="/tmp/juggling-native-transport-check-"+std::to_string(getpid());
 try{
  auto small=std::make_unique<SpscQueue<RawMarker,4>>();
  for(unsigned k=0;k<4;++k){RawMarker marker;marker.id=k;require(small->push(marker),"Premature queue overflow");}
  require(!small->push(RawMarker{}),"Queue failed to detect overflow");
  for(unsigned k=0;k<4;++k){RawMarker marker;require(small->pop(marker)&&marker.id==k,"Queue ordering failed");}
  RawMarker marker;require(!small->pop(marker),"Empty queue returned data");
  auto region=SharedRegion::create(path);auto& state=region.get();
  bool rejected=false;try{auto duplicate=SharedRegion::create(path);}catch(const std::exception&){rejected=true;}
  require(rejected,"Existing shared-memory file was overwritten");
  state.magic=0;rejected=false;try{auto invalid=SharedRegion::open(path);}catch(const std::exception&){rejected=true;}
  require(rejected,"Invalid shared-memory protocol accepted");state.magic=robot_magic;
  ViconPacket packet;validate(packet);packet.unlabeled_count=513;rejected=false;
  try{validate(packet);}catch(const std::exception&){rejected=true;}require(rejected,"Vicon overflow accepted");
  constexpr unsigned count=20000;
  const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
  pid_t child=fork();require(child>=0,"fork failed");
  if(child==0){
   try{
    auto remote=SharedRegion::open(path);auto& channel=remote.get();
    for(unsigned k=0;k<count;++k){SessionCommand command;
     while(!channel.commands.pop(command)){
      if(std::chrono::steady_clock::now()>deadline)_exit(2);std::this_thread::yield();}
     if(command.id!=static_cast<int>(k)||command.goal[6]!=k*.5||command.trajectory[111]!=k*.25)_exit(3);
     NativePacket response;response.move_id=command.id;response.q[6]=command.goal[6];response.reference_q_dq_ddq[20]=command.trajectory[111];
     while(!channel.states.push(response)){
      if(std::chrono::steady_clock::now()>deadline)_exit(4);std::this_thread::yield();}
    }
    channel.producer_state.store(2,std::memory_order_release);_exit(0);
   }catch(...){_exit(5);}
  }
  unsigned sent=0,received=0;
  while(received<count){
   if(std::chrono::steady_clock::now()>deadline){kill(child,SIGKILL);int status;waitpid(child,&status,0);throw std::runtime_error("Cross-process transport timeout");}
   if(sent<count){SessionCommand command;command.kind=SessionCommand::Move;command.id=sent;command.goal[6]=sent*.5;command.trajectory[111]=sent*.25;
    if(state.commands.push(command))++sent;}
   NativePacket response;
   if(state.states.pop(response)){
    require(response.move_id==static_cast<int>(received)&&response.q[6]==received*.5&&response.reference_q_dq_ddq[20]==received*.25,"Cross-process data/order mismatch");++received;
   }
  }
  int status=0;require(waitpid(child,&status,0)==child&&WIFEXITED(status)&&WEXITSTATUS(status)==0,"Transport child failed");
  require(state.producer_state.load(std::memory_order_acquire)==2,"Cross-process status not visible");
  std::filesystem::remove(path);
  std::cout<<"{\"passed\":true,\"cross_process_packets\":"<<count<<",\"robot_packet_bytes\":"<<sizeof(NativePacket)<<",\"vicon_packet_bytes\":"<<sizeof(ViconPacket)<<"}\n";return 0;
 }catch(const std::exception& error){std::filesystem::remove(path);std::cerr<<error.what()<<'\n';return 1;}
}
