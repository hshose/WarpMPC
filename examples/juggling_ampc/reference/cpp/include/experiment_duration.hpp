#pragma once
namespace juggling {
// Zero requests operator/guard-controlled termination, with no elapsed cutoff.
inline bool experiment_duration_expired(double elapsed,double seconds){
 return seconds>0. && elapsed>=seconds;
}
}
