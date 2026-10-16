#include "c_api.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

static void stats(std::vector<double> v,const char* label){
 std::sort(v.begin(),v.end());double at=.95*(v.size()-1);size_t i=static_cast<size_t>(at);
 double p95=v[i]+(at-i)*(v[std::min(i+1,v.size()-1)]-v[i]);double median=.5*(v[(v.size()-1)/2]+v[v.size()/2]);
 std::cout<<"\""<<label<<"\":{\"count\":"<<v.size()<<",\"median_ms\":"<<median<<",\"mean_ms\":"<<std::accumulate(v.begin(),v.end(),0.)/v.size()<<",\"p95_ms\":"<<p95<<",\"max_ms\":"<<v.back()<<"}";
}
int main(int argc,char** argv){
 if(argc<3){std::cerr<<"Usage: juggling_benchmark FACTOR_DIR replay.csv [output.csv] [repetitions=1] [compiled=1]\n";return 2;}
 std::ifstream input(argv[2]);if(!input){std::cerr<<"Cannot open replay\n";return 2;}
 std::vector<std::array<double,31>> rows;std::string line;
 while(std::getline(input,line)){if(line.empty()||line[0]=='#')continue;std::replace(line.begin(),line.end(),',',' ');std::istringstream in(line);std::array<double,31> row;for(double& v:row)if(!(in>>v)){std::cerr<<"Invalid replay row\n";return 2;}rows.push_back(row);}
 if(rows.size()<2)return 2;
 std::ofstream output(argc>3?argv[3]:"benchmark_steps.csv");output<<"repetition,index,simulation_time,total_ms,shift_ms,data_ms,qp_ms,merit_ms,iterations,alpha,violation,clipping\n";output<<std::setprecision(17);
 int repetitions=argc>4?std::stoi(argv[4]):1,compiled=argc>5?std::stoi(argv[5]):1;
 if(repetitions<1||repetitions>1000)return 2;
 std::vector<double> total,shift,data,qp,merit,first;int fallbacks=0;char error[2048]{};std::vector<double> plan;std::array<double,7> target{};std::array<double,18> m{};
 for(int repetition=0;repetition<repetitions;++repetition){void* controller=juggling_create(argv[1],compiled,error,sizeof(error));if(!controller){std::cerr<<error<<'\n';return 1;}
  int dimensions[4];juggling_dimensions(controller,dimensions);plan.resize(dimensions[0]);
  for(size_t i=0;i<rows.size();++i){const auto& row=rows[i];double dt=i?row[0]-rows[i-1][0]:0.;
   if(juggling_step(controller,row.data()+1,row.data()+24,dt,plan.data(),target.data(),m.data(),error,sizeof(error))){std::cerr<<"Step "<<i<<": "<<error<<'\n';juggling_destroy(controller);return 1;}
   if(m[12]!=50){std::cerr<<"Iteration budget changed\n";return 1;}
   output<<repetition<<','<<i<<','<<row[0];for(int j=0;j<5;++j)output<<','<<m[j];output<<','<<m[12]<<','<<m[8]<<','<<m[6]<<','<<m[11]<<'\n';
   if(i){total.push_back(m[0]);shift.push_back(m[1]);data.push_back(m[2]);qp.push_back(m[3]);merit.push_back(m[4]);}else first.push_back(m[0]);if(m[15]>=0)++fallbacks;
  }juggling_destroy(controller);
 }
 std::cout<<std::setprecision(10)<<"{";stats(total,"steady_total");std::cout<<',';stats(first,"first_call");std::cout<<',';stats(shift,"warmstart");std::cout<<',';stats(data,"data");std::cout<<',';stats(qp,"qp");std::cout<<',';stats(merit,"merit");std::cout<<",\"evaluator_threads\":"<<m[16]<<",\"fallbacks\":"<<fallbacks<<",\"compiled_factors\":"<<(compiled?"true":"false")<<"}\n";
}
