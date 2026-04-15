namespace go idl.labcv.gateway.vbase
namespace py idl.labcv.gateway.vbase
namespace cpp idl.labcv.gateway.vbase

struct VBaseResp {
  1: i32 timeElapsed, // in ms
  2: i32 receivedAt,  // timestamp (ms)
  3: i32 sentAt,      // timestamp (ms)
}