include "../../base.thrift"

namespace go ic.cv
namespace py ic.cv
namespace cpp ic.cv

struct AlgoServiceReq {
  1: list<binary> binary_data,
  2: string req_json,
  3: optional string req_key, // to support one service with multiple abilities
  255: optional base.Base Base,
}

struct AlgoServiceResp {
  1: list<binary> binary_data,
  2: string resp_json,
  255: optional base.BaseResp BaseResp,
}

service AlgoService {
    AlgoServiceResp Process(1: AlgoServiceReq req),
}
