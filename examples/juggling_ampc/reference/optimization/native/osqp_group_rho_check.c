/* Tiny offline-only regression; no controller, rollout, or robot dependency. */
#include "osqp_group_rho.h"
#include "types.h"
#include "algebra_vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double worst_kkt=0.,worst_public=0.;
static int installations=0;
static void require(int ok,const char* what){if(!ok){fprintf(stderr,"group-rho check: %s\n",what);exit(1);}}
static OSQPSolver* setup(int m,const double* p,const double* a,const double* l,const double* u,double rho,int iterations){
  /* A rows [x, y, x+y, x-y], last row omitted for the all-inequality case. */
  OSQPInt pi[]={0,1,2},pr[]={0,1},ai4[]={0,3,6},ar4[]={0,2,3,1,2,3},ai3[]={0,2,4},ar3[]={0,2,1,2};
  double q[]={-1.,-3.};OSQPSettings settings;OSQPSolver* solver=NULL;
  OSQPCscMatrix* P=OSQPCscMatrix_new(2,2,2,(OSQPFloat*)p,pr,pi);
  OSQPCscMatrix* A=OSQPCscMatrix_new(m,2,m==4?6:4,(OSQPFloat*)a,m==4?ar4:ar3,m==4?ai4:ai3);
  osqp_set_default_settings(&settings);settings.verbose=0;settings.scaling=0;settings.rho=rho;
  settings.rho_is_vec=1;settings.adaptive_rho=OSQP_ADAPTIVE_RHO_UPDATE_DISABLED;
  settings.max_iter=iterations;settings.check_termination=0;settings.polishing=0;settings.warm_starting=1;
  require(!osqp_setup(&solver,P,q,A,l,u,m,2,&settings),"setup");OSQPCscMatrix_free(P);OSQPCscMatrix_free(A);
  return solver;
}
static double kkt_error(OSQPSolver* solver,const double* p,const double* a){
  /* Direct callback solves the KKT and returns [x_tilde; A*x_tilde]. Compare
   * an analytic 2x2 Schur solve using WORKSPACE rho: this detects stale cached
   * KKT diagonals AND stale linear-solver reciprocals, without QDLDL layout. */
  int i,m=solver->work->data->m;double rho[4],dense[4][2]={{0}},rhs[]={.7,-.4,.2,-.1,.3,-.2};
  OSQPVectorf* b=OSQPVectorf_new(rhs,m+2);OSQPVectorf_to_raw(rho,solver->work->rho_vec);
  dense[0][0]=a[0];dense[2][0]=a[1];
  if(m==4){dense[3][0]=a[2];dense[1][1]=a[3];dense[2][1]=a[4];dense[3][1]=a[5];}
  else{dense[1][1]=a[2];dense[2][1]=a[3];}
  double h00=p[0]+solver->settings->sigma,h11=p[1]+solver->settings->sigma,h01=0.,r0=rhs[0],r1=rhs[1],error=0.;
  for(i=0;i<m;++i){const double x=dense[i][0],y=dense[i][1];
    h00+=rho[i]*x*x;h11+=rho[i]*y*y;h01+=rho[i]*x*y;r0+=rho[i]*x*rhs[i+2];r1+=rho[i]*y*rhs[i+2];}
  const double determinant=h00*h11-h01*h01,x=(r0*h11-r1*h01)/determinant,y=(r1*h00-r0*h01)/determinant;
  require(!solver->work->linsys_solver->solve(solver->work->linsys_solver,b,1),"direct KKT solve");
  const double* actual=OSQPVectorf_data(b);error=fmax(fabs(actual[0]-x),fabs(actual[1]-y));
  for(i=0;i<m;++i)error=fmax(error,fabs(actual[i+2]-dense[i][0]*x-dense[i][1]*y));
  OSQPVectorf_free(b);return error;
}
static void install(OSQPSolver* solver,const unsigned char* selected,double multiplier,const double* p,const double* a){
  int i,m=solver->work->data->m;OSQPInt types[4];double scratch[4],actual[4],inverse[4];JugglingGroupRhoReport report;
  require(!juggling_osqp_install_group_rho(solver,m,selected,multiplier,types,scratch,&report),"bridge installation");
  OSQPVectorf_to_raw(actual,solver->work->rho_vec);OSQPVectorf_to_raw(inverse,solver->work->rho_inv_vec);
  for(i=0;i<m;++i){double expected=types[i]<0?OSQP_RHO_MIN:(types[i]>0?1000.*solver->settings->rho:solver->settings->rho*(selected[i]?multiplier:1.));
    require(actual[i]==expected,"inequality/equality/free penalty policy");
    require(fabs(inverse[i]*actual[i]-1.)<1e-14,"workspace reciprocal");}
  double error=kkt_error(solver,p,a);worst_kkt=fmax(worst_kkt,error);require(error<2e-10,"cached KKT/reciprocal consistency");
  ++installations;
}
static void exact_noop(OSQPSolver* solver){
  const int m=solver->work->data->m;double before[8],after[8];
  OSQPVectorf_to_raw(before,solver->work->rho_vec);OSQPVectorf_to_raw(before+m,solver->work->rho_inv_vec);
  require(!juggling_osqp_install_group_rho(solver,0,NULL,1.,NULL,NULL,NULL),"multiplier-one bypass");
  OSQPVectorf_to_raw(after,solver->work->rho_vec);OSQPVectorf_to_raw(after+m,solver->work->rho_inv_vec);
  require(!memcmp(before,after,2*m*sizeof(double)),"multiplier-one changed workspace");
}
static void mixed_rows(void){
  const unsigned char selected[]={1,1,1,0};double p[]={2.,4.},a[]={1.,1.,1.,1.,1.,-1.};
  double l[]={-1.,.2,-OSQP_INFTY,-.4},u[]={.4,.2,OSQP_INFTY,.8};
  OSQPSolver* solver=setup(4,p,a,l,u,.001,37);exact_noop(solver);
  install(solver,selected,10.,p,a);exact_noop(solver);
  for(int repeat=0;repeat<4;++repeat){
    if(repeat==1){l[1]=-.3;u[1]=.7;l[2]=u[2]=.1;} // equality->inequality and free->equality
    if(repeat==2){l[0]=-OSQP_INFTY;u[0]=OSQP_INFTY;} // selected inequality->free
    if(repeat==3){l[0]=-.8;u[0]=.6;l[2]=-OSQP_INFTY;u[2]=OSQP_INFTY;}
    p[0]+=.1;p[1]+=.2;a[1]+=.03;
    require(!osqp_update_data_vec(solver,NULL,l,u),"bounds update");
    require(!osqp_update_data_mat(solver,p,NULL,2,a,NULL,6),"matrix update");
    /* First update leaves classifications unchanged, exposing the overwrite
     * bug this bridge fixes: workspace defaults versus cached custom KKT. */
    if(repeat==0)require(kkt_error(solver,p,a)>1e-7,"regression did not expose stale penalty/KKT mismatch");
    install(solver,selected,repeat%2?10.:30.,p,a);exact_noop(solver);
  }
  osqp_cleanup(solver);
}
static void compare_public(double multiplier,int updated){
  const unsigned char selected[]={1,1,1};double p[]={2.,4.},a[]={1.,1.,1.,1.};
  double l[]={-1.,-.5,-.3},u[]={.4,.7,.8};
  OSQPSolver* custom=setup(3,p,a,l,u,.001,37);install(custom,selected,multiplier,p,a);
  if(updated){p[0]=2.3;p[1]=3.7;a[1]=.8;a[3]=1.2;u[0]=.3;u[1]=.6;
    require(!osqp_update_data_vec(custom,NULL,l,u),"comparison bounds update");
    require(!osqp_update_data_mat(custom,p,NULL,2,a,NULL,4),"comparison matrix update");install(custom,selected,multiplier,p,a);}
  OSQPSolver* reference=setup(3,p,a,l,u,.001*multiplier,37);
  osqp_cold_start(custom);osqp_cold_start(reference);
  require(!osqp_solve(custom)&&!osqp_solve(reference),"fixed-iteration comparison solve");
  for(int i=0;i<2;++i)worst_public=fmax(worst_public,fabs(custom->solution->x[i]-reference->solution->x[i]));
  for(int i=0;i<3;++i)worst_public=fmax(worst_public,fabs(custom->solution->y[i]-reference->solution->y[i]));
  require(worst_public<2e-10,"custom inequalities differ from public scalar rho");
  double before[14],after[14];
  OSQPVectorf_to_raw(before,custom->work->rho_vec);OSQPVectorf_to_raw(before+3,custom->work->rho_inv_vec);
  OSQPVectorf_to_raw(before+6,custom->work->x);OSQPVectorf_to_raw(before+8,custom->work->y);OSQPVectorf_to_raw(before+11,custom->work->z);
  require(!osqp_update_data_vec(custom,NULL,NULL,NULL),"continuation information reset");
  OSQPVectorf_to_raw(after,custom->work->rho_vec);OSQPVectorf_to_raw(after+3,custom->work->rho_inv_vec);
  OSQPVectorf_to_raw(after+6,custom->work->x);OSQPVectorf_to_raw(after+8,custom->work->y);OSQPVectorf_to_raw(after+11,custom->work->z);
  require(!memcmp(before,after,sizeof(before)),"exact continuation changed penalties/ADMM state");
  /* No bridge, refactor, or warm start on the second 37-iteration call. */
  require(!osqp_solve(custom),"exact continuation solve");osqp_cleanup(reference);
  reference=setup(3,p,a,l,u,.001*multiplier,74);osqp_cold_start(reference);require(!osqp_solve(reference),"public 74-iteration solve");
  for(int i=0;i<2;++i)worst_public=fmax(worst_public,fabs(custom->solution->x[i]-reference->solution->x[i]));
  for(int i=0;i<3;++i)worst_public=fmax(worst_public,fabs(custom->solution->y[i]-reference->solution->y[i]));
  require(worst_public<2e-10,"continued ADMM differs from uninterrupted public solve");
  osqp_cleanup(custom);osqp_cleanup(reference);
}
int main(void){
  require(!juggling_osqp_install_group_rho(NULL,0,NULL,1.,NULL,NULL,NULL),"null-solver default bypass");
  mixed_rows();compare_public(10.,0);compare_public(30.,1);
  printf("{\"passed\":true,\"installations\":%d,\"kkt_max_abs_error\":%.17g,\"public_solver_max_abs_error\":%.17g,\"default_exact_bypass\":true,\"repeated_updates_and_type_changes\":true,\"exact_continuation\":true}\n",installations,worst_kkt,worst_public);
  return 0;
}
