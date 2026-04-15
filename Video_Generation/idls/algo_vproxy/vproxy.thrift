include "common.thrift"

namespace go idl.labcv.gateway.vproxy
namespace py idl.labcv.gateway.vproxy
namespace cpp idl.labcv.gateway.vproxy

service VisionService {
  //// Business Side Use
  // sync
  common.AlgoResp Process(1: common.AlgoReq req),

  // async
  common.SubmitTaskResp SubmitTask(1: common.SubmitTaskReq req),
  common.GetResultResp GetResult(1: common.GetResultReq req),

  //// Internal Use
  // update task
  common.UpdateTaskResp UpdateTask(1: common.UpdateTaskReq req),
  // callback func template
  common.TaskCallbackResp DoTaskCallback(1: common.TaskCallbackReq req),
}
