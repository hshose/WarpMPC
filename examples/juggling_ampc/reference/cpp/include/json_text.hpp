#pragma once
#include <string>

namespace juggling {
// Exception messages can contain newlines; keep saved failure records valid JSON.
inline std::string json_text(const std::string& value){
  constexpr char hex[]="0123456789abcdef";
  std::string out="\"";
  for(unsigned char c:value){
    if(c=='\"'||c=='\\'){out+='\\';out+=static_cast<char>(c);}
    else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
    else out+=static_cast<char>(c);
  }
  return out+'\"';
}
}
