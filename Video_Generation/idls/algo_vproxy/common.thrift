include "vbase.thrift"
include "../base.thrift"

namespace go  idl.labcv.gateway.common
namespace py  idl.labcv.gateway.common
namespace cpp idl.labcv.gateway.common

// 鉴权参数
struct AuthInfo {
  1: string app_key,
  2: string timestamp,
  3: string nonce,
  4: string sign, // 使用app_secret+nonce+timestamp生成的签名,app_secret为应用秘钥,从智创控制台获取
}
// 同步接口请求结构体
struct AlgoReq {
  1: string req_key,
  2: list<binary> binary_data,
  3: string req_json,
  4: optional binary req_custom_structure,
  5: AuthInfo auth_info,
  255: optional base.Base Base,
}

struct AlgoResp {
  1: list<binary> binary_data,
  2: string resp_json,
  254: vbase.VBaseResp vbase,
  255: optional base.BaseResp BaseResp,
}

// Async
enum CallbackType {
  RPC = 0 // need psm, idc, cluster
  HTTP = 1 // need http_url
  EVENTBUS = 2
}

struct TaskCallbackParam {
  1: CallbackType callback_type,
  2: optional string http_url,
  3: optional string psm,
  4: optional string idc,
  5: optional string cluster,
  6: optional string event
}

struct SubmitTaskReq {
  1: string req_key,
  2: list<binary> binary_data,
  3: string req_json,
  4: optional TaskCallbackParam task_callback_param,
  5: AuthInfo auth_info,
  255: optional base.Base Base,
}

struct SubmitTaskResp {
  1: string task_id,
  2: string resp_json,
  254: vbase.VBaseResp vbase,
  255: optional base.BaseResp BaseResp,
}

struct GetResultReq {
  1: string req_key,
  2: string task_id,
  3: string req_json,
  255: optional base.Base Base,
}

struct GetResultResp {
  1: list<binary> binary_data,
  2: string resp_json,
  3: string status,
  254: vbase.VBaseResp vbase,
  255: optional base.BaseResp BaseResp,
}

// Task Status Update
struct UpdateTaskReq {
  1: string task_id,
  2: string req_json,
  255: optional base.Base Base,
}

struct UpdateTaskResp {
  1: string resp_json,
  254: vbase.VBaseResp vbase,
  255: optional base.BaseResp BaseResp,
}

// Callback Template
struct TaskCallbackReq {
  1: string task_id,
  2: optional string req_json, // 预留字段，额外返回信息
  255: optional base.Base Base,
}

struct TaskCallbackResp {
  1: optional string resp_json, // 预留字段，额外返回信息
  255: optional base.BaseResp BaseResp, // statusCode = 0为成功，其余为失败
}