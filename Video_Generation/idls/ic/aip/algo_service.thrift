include "../../base.thrift"

namespace go IC.AIP
namespace py IC.AIP
namespace cpp IC.AIP

// 计量信息，详细设计见：https://bytedance.feishu.cn/docx/doxcnYptmELIKxX0MiDx6KT09Re
struct ReqMeasureInfo {
  1: string measure_type, // 计量类型
  2: i64 value // 计量数量
}

struct AlgoServiceReq {
  1: list<binary> binary_data,
  2: string req_json,
  3: optional string req_key, // to support one service with multiple abilities
  4: optional binary req_custom_structure,
  5: string algorithm_key, // algorithm_key indicates one algorithm in this service, might be same as req_key
  255: optional base.Base Base,
}

struct AlgoServiceResp {
  1: list<binary> binary_data,
  2: string resp_json,
  3: optional binary resp_custom_structure,
  254: optional ReqMeasureInfo req_measure_info,
  255: optional base.BaseResp BaseResp,
}

service AlgoService {
    AlgoServiceResp Process(1: AlgoServiceReq req),
}

