#pragma once
// Non-realtime experiment geometry. All coordinates are in robot-base axes.
#include "../../optimization/native/controller.hpp"
#include "model_config.hpp"
#include "json_text.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <Eigen/Geometry>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>

namespace juggling::experiment {
namespace g=generated;
namespace pt=boost::property_tree;
using Pose=Eigen::Matrix4d;
inline pt::ptree read_json(const std::filesystem::path& p) {
 // Legacy factor metadata was emitted by Python with +/-Infinity in bound
 // arrays. Preserve these as strings for the strict parser; finite model and
 // calibration scalars still undergo explicit numeric validation.
 std::ifstream f(p);if(!f)throw std::runtime_error("Cannot read "+p.string());std::string input((std::istreambuf_iterator<char>(f)),{}),clean;clean.reserve(input.size());bool quoted=false,escape=false;
 for(size_t i=0;i<input.size();){char c=input[i];if(quoted){clean+=c;++i;if(escape)escape=false;else if(c=='\\')escape=true;else if(c=='"')quoted=false;continue;}
  if(c=='"'){quoted=true;clean+=c;++i;continue;}size_t n=input.compare(i,9,"-Infinity")==0?9:input.compare(i,8,"Infinity")==0?8:input.compare(i,3,"NaN")==0?3:0;
  if(n){clean+='"';clean.append(input,i,n);clean+='"';i+=n;}else{clean+=c;++i;}}
 pt::ptree v;std::istringstream stream(clean);pt::read_json(stream,v);return v;
}
template<int N> Eigen::Matrix<double,N,1> vector(const pt::ptree& a){
 Eigen::Matrix<double,N,1> x;int i=0;for(const auto& v:a){if(i==N)throw std::runtime_error("Oversized configuration vector");x[i++]=v.second.get_value<double>();}
 if(i!=N||!x.allFinite())throw std::runtime_error("Invalid configuration vector");return x;
}
template<size_t N> Eigen::Map<const Eigen::Matrix<double,N,1>> map(const std::array<double,N>& a){return Eigen::Map<const Eigen::Matrix<double,N,1>>(a.data());}
inline Pose pose(const pt::ptree& a){Pose x;int i=0;for(const auto& row:a){if(i==4)throw std::runtime_error("Invalid pose rows");x.row(i++)=vector<4>(row.second).transpose();}
 if(i!=4||!x.allFinite()||(x.row(3)-Eigen::RowVector4d(0,0,0,1)).norm()>1e-8||
 (x.topLeftCorner<3,3>().transpose()*x.topLeftCorner<3,3>()-Eigen::Matrix3d::Identity()).norm()>1e-5||std::abs(x.topLeftCorner<3,3>().determinant()-1)>1e-5)throw std::runtime_error("Invalid rigid transform");return x;}
template<class Derived> std::string json_matrix(const Eigen::MatrixBase<Derived>& a){std::ostringstream s;s<<std::setprecision(17)<<'[';
 for(int i=0;i<a.rows();++i){if(i)s<<',';s<<'[';for(int j=0;j<a.cols();++j){if(j)s<<',';if(std::isfinite(a(i,j)))s<<a(i,j);else s<<"null";}s<<']';}s<<']';return s.str();}
template<class Derived> std::string json_vector(const Eigen::MatrixBase<Derived>& a){auto evaluated=a.eval();std::ostringstream s;s<<std::setprecision(17)<<'[';for(int i=0;i<evaluated.size();++i){if(i)s<<',';if(std::isfinite(evaluated.data()[i]))s<<evaluated.data()[i];else s<<"null";}s<<']';return s.str();}
inline void write(const std::filesystem::path& p,const std::string& s){std::ofstream f(p,std::ios::out|std::ios::trunc);f<<s<<'\n';f.close();if(!f)throw std::runtime_error("Cannot save "+p.string());}
inline double median(std::vector<double> v){if(v.empty())throw std::runtime_error("Empty median");std::sort(v.begin(),v.end());size_t n=v.size();return n%2?v[n/2]:.5*(v[n/2-1]+v[n/2]);}
inline double angle(const Eigen::Matrix3d& R){return Eigen::AngleAxisd(R).angle();}
// Robot-base +X is forward. Extend the forward and lateral semi-axes while
// preserving the previous rearward limits and a rounded diagonal boundary.
inline bool within_landing_reach(const Eigen::Vector2d& hit,const Eigen::Vector2d& fixture){
 auto inside=[](const Eigen::Vector2d& p,double front,double side,double rear){
  const double x=p.x()/(p.x()>=0?front:rear),y=p.y()/side;return x*x+y*y<=1.;};
 return inside(hit,g::landing_forward,g::landing_side,.35)&&
        (g::fixture_landing_forward<=0||inside(hit-fixture,g::fixture_landing_forward,g::fixture_landing_side,.12));
}
inline double nominal_return_fraction(double time,double first_impact,double duration){
 if(duration<=0||!std::isfinite(first_impact))return 0.;
 const double u=std::clamp((time-first_impact)/duration,0.,1.);
 return u*u*u*(10.+u*(-15.+6.*u));
}
struct Pair {Pose base_flange,vicon_paddle;double capture=0,robot=0;};
inline Pose align(const Pose& mount,const std::vector<Pair>& pairs){
 if(pairs.size()<30)throw std::runtime_error("Need 30 settled Vicon/robot pose pairs");
 Eigen::Matrix4d accumulator=Eigen::Matrix4d::Zero();
 for(const auto& p:pairs){Pose Y=p.vicon_paddle*(p.base_flange*mount).inverse();Eigen::Quaterniond q(Y.topLeftCorner<3,3>());accumulator+=q.coeffs()*q.coeffs().transpose();}
 Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> eig(accumulator);Eigen::Quaterniond q;q.coeffs()=eig.eigenvectors().col(3);
 Pose Y=Pose::Identity();Y.topLeftCorner<3,3>()=q.normalized().toRotationMatrix();
 for(int j=0;j<3;++j){std::vector<double> t;for(const auto& p:pairs){Eigen::Vector3d delta=p.vicon_paddle.topRightCorner<3,1>()-Y.topLeftCorner<3,3>()*(p.base_flange*mount).topRightCorner<3,1>();t.push_back(delta[j]);}Y(j,3)=median(t);}
 for(const auto& p:pairs){Pose predicted=Y*p.base_flange*mount;
  if((predicted.topRightCorner<3,1>()-p.vicon_paddle.topRightCorner<3,1>()).norm()>.003||angle(predicted.topLeftCorner<3,3>().transpose()*p.vicon_paddle.topLeftCorner<3,3>())>.5*M_PI/180)throw std::runtime_error("World alignment is not stable (3 mm / 0.5 degree)");}
 if(std::acos(std::clamp(Y(2,2),-1.,1.))>3*M_PI/180)throw std::runtime_error("Robot base/Vicon up differ by more than 3 degrees");return Y;
}
inline Joint fixture_home(RobotModel& model,const Eigen::Vector3d& ball,double offset,const Eigen::Matrix3d& normal_rotation){
 Eigen::Vector3d target=map(g::origin);target.head<2>()=ball.head<2>();target[2]+=offset;
 const Eigen::Vector2d xy=(target-map(g::origin)).head<2>();
 if(!within_landing_reach(xy,xy)||!ball.allFinite()||!std::isfinite(offset)||std::abs(offset)>.05)throw std::runtime_error("Fixture is outside startup workspace");
 double gap=ball[2]-target[2]-g::radius;if(gap<1.)throw std::runtime_error("Ball bottom must be at least 1.00 m above strike plane; measured gap="+std::to_string(gap)+" m");
 Eigen::Matrix3d nominal=model.contact_fk(map(g::home))[4]*normal_rotation.transpose();
 Joint q=map(g::home),lo=map(g::joint_lower).cwiseMax((q.array()-1.).matrix()),hi=map(g::joint_upper).cwiseMin((q.array()+1.).matrix());
 auto residual=[&](const Joint& at){Eigen::Matrix<double,19,1> r;auto f=model.contact_fk(at);r.head<3>()=(f[0]-target)/.001;
  Eigen::Matrix3d dr=(f[4]-nominal)/.01;r.segment<9>(3)=Eigen::Map<Eigen::Matrix<double,9,1>>(dr.data());r.tail<7>()=.1*(at-map(g::home));return r;};
 double lambda=1e-3;
 for(int k=0;k<100;++k){auto r=residual(q);Eigen::Matrix<double,19,7> J;
  for(int j=0;j<7;++j){Joint plus=q,minus=q;plus[j]+=1e-5;minus[j]-=1e-5;J.col(j)=(residual(plus)-residual(minus))/2e-5;}
  Joint step=-(J.transpose()*J+lambda*Eigen::Matrix<double,7,7>::Identity()).ldlt().solve(J.transpose()*r);
  if(!step.allFinite())throw std::runtime_error("Fixture IK failed");if(step.norm()<1e-10)break;
  Joint trial=(q+step).cwiseMax(lo).cwiseMin(hi);if(residual(trial).squaredNorm()<r.squaredNorm()){q=trial;lambda=std::max(1e-8,lambda*.3);}else lambda*=10;
 }
 auto f=model.contact_fk(q);if((f[0]-target).norm()>.001||(f[1]-Eigen::Vector3d::UnitZ()).norm()>.01)throw std::runtime_error("Cannot level paddle below fixture inside joint envelope");return q;
}
struct HomeReference {Eigen::Vector3d offset=Eigen::Vector3d::Zero();Joint posture=map(g::home);};
inline HomeReference configured_home(RobotModel& model,const std::filesystem::path& config,const Eigen::Matrix3d& normal_rotation){
 HomeReference home;if(config.empty())return home;
 auto settings=read_json(config);if(settings.get<int>("schema_version")!=1)throw std::runtime_error("Unsupported juggling home configuration");
 Eigen::Vector3d face=map(g::origin);face.x()=settings.get<double>("x_base_m");
 face.y()=settings.get<double>("y_base_m",face.y());face.z()=settings.get<double>("z_base_m",face.z());
 home.offset=face-map(g::origin);Eigen::Vector3d synthetic_ball=face;synthetic_ball.z()+=1.1+g::radius;
 home.posture=fixture_home(model,synthetic_ball,home.offset.z(),normal_rotation);return home;
}
inline Eigen::Matrix<double,4,28,Eigen::RowMajor> trajectory(RobotModel& model,const Matrix& Y,const Joint& q,const Joint& dq,const Joint& initial_acceleration,double& raw,double& projection){
 if(Y.rows()<4||Y.cols()!=34||!Y.allFinite())throw std::runtime_error("MPC plan must contain rows of 34 finite physical variables");
 Eigen::Matrix<double,4,7> a,ap;raw=0;projection=0;
 for(int k=0;k<4;++k){Joint qk=Y.row(k).segment<7>(6),vk=Y.row(k).segment<7>(13),tau=Y.row(k).segment<7>(23);a.row(k)=model.mass(qk).ldlt().solve(tau-model.bias(qk,vk)).transpose();}
 if(!a.allFinite()||!initial_acceleration.allFinite())throw std::runtime_error("Nonfinite MPC acceleration");
 for(int k=0;k<4;++k)for(int j=0;j<7;++j){if(k==1||k==2)raw=std::max(raw,std::abs(a(k,j))/g::acceleration_limit[j]);ap(k,j)=std::clamp(a(k,j),-.95*g::acceleration_limit[j],.95*g::acceleration_limit[j]);if(k==1||k==2)projection=std::max(projection,std::abs(ap(k,j)-a(k,j))/g::acceleration_limit[j]);}
 if(raw>1.25)throw std::runtime_error("MPC acceleration defect exceeds reference projection budget");ap.row(0)=initial_acceleration.transpose();
 Eigen::Matrix<double,4,28,Eigen::RowMajor> T=Eigen::Matrix<double,4,28,Eigen::RowMajor>::Zero();T.row(0).head<7>()=q;T.row(0).segment<7>(7)=dq;
 for(int k=0;k<4;++k){T.row(k).tail<7>()=ap.row(k);T.row(k).segment<7>(14)=model.rnea(T.row(k).head<7>(),T.row(k).segment<7>(7),ap.row(k)).transpose();
  if(k<3){double h=g::stage_steps[k];T.row(k+1).head<7>()=T.row(k).head<7>()+h*T.row(k).segment<7>(7)+h*h*(2*ap.row(k)+ap.row(k+1))/6;T.row(k+1).segment<7>(7)=T.row(k).segment<7>(7)+.5*h*(ap.row(k)+ap.row(k+1));}}
 return T;
}
inline Eigen::Vector2d landing(const State& x,const Eigen::Vector3d& gravity,double drag,double plane){
 if(!x.allFinite()||gravity[2]>=0||drag<0)throw std::runtime_error("Invalid landing state");if(x[2]<=plane)return x.head<2>();
 auto position=[&](double t){double a=drag>1e-8?-std::expm1(-drag*t)/drag:t;double b=drag>1e-8?(t-a)/drag:.5*t*t;return Eigen::Vector3d(x.head<3>()+a*x.segment<3>(3)+b*gravity);};
 double lo=0,hi=1;while(position(hi)[2]>plane){hi*=2;if(hi>64)throw std::runtime_error("No strike-plane crossing");}
 for(int k=0;k<40;++k){double mid=.5*(lo+hi);if(position(mid)[2]>plane)lo=mid;else hi=mid;}return position(.5*(lo+hi)).head<2>();
}
}
