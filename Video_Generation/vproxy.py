import euler
import thriftpy2
from euler.base_compat_middleware import client_middleware
import os
import bytedenv
import bytedance.context
from bytedance import metrics
import time
from euler import errors
from bytedance.ivory.errno import error_code

current_dir = os.path.dirname(os.path.abspath(__file__))
mcli = metrics.Client(prefix="atomu_common_entrance")
base_thrift = thriftpy2.load(current_dir + "/idls/base.thrift", module_name="base_thrift")
gateway_common_thrift = thriftpy2.load(
    current_dir + "/idls/algo_vproxy/common.thrift", module_name="common_thrift"
)
gateway_vproxy_thrift = thriftpy2.load(
    current_dir + "/idls/algo_vproxy/vproxy.thrift", module_name="vproxy_thrift"
)

DEFAULT_APP_KEY = "a33a6b0687584a7eaac1f7109710b018"


class VproxyClient:
    def __init__(self):
        GATEWAY_TARGET = os.getenv(
            "VPROXY_GATEWAY_TARGET", "sd://toutiao.labcv.algo_vproxy?cluster=default"
        )
        if bytedenv.get_idc_name() in ["uswest2", "fr1a", "ussw1a", "ussw1b"]:
            GATEWAY_TARGET = os.getenv(
                "VPROXY_GATEWAY_TARGET",
                "sd://toutiao.labcv.algo_vproxy?cluster=default&idc=maliva",
            )
        self.gateway_client = euler.Client(
            service_cls=gateway_vproxy_thrift.VisionService,
            target=GATEWAY_TARGET,
            timeout=120,
            transport="ttheader",
        )
        # 兜底algorithm_key为空时mesh_middleware crash
        if bytedance.context.get("algorithm_key") == None:
            bytedance.context.set("algorithm_key", "default")
        if bytedance.context.get("IC_APP_KEY") is None:
            bytedance.context.set("IC_APP_KEY", DEFAULT_APP_KEY)

        self.gateway_client.use(client_middleware)

    def process(self, req_key: str, req_json: str, binary_data: list):
        req = gateway_common_thrift.AlgoReq()
        req.req_key = req_key
        req.binary_data = []
        req.req_json = req_json
        req.binary_data = binary_data

        st = time.time()
        resp = self.gateway_client.Process(req)
        et = time.time()
        mcli.emit_counter("rpc.throughput", 1, tags={"req_key": req_key})
        mcli.emit_timer("rpc.latency", (et - st) * 1000, tags={"req_key": req_key})

        if resp.BaseResp.StatusCode == 10000 or resp.BaseResp.StatusCode == 10001:
            raise TimeoutError(
                f"[Timeout]Failed to call gateway service. req_key: {req_key}, StatusCode: {resp.BaseResp.StatusCode}, StatusMessage: {resp.BaseResp.StatusMessage}"
            )

        if resp.BaseResp.StatusCode == 11001:
            raise errors.EulerError(
                error_code.DownstreamConFailedError,
                f"[Connection Error]Failed to connect service. req_key: {req_key}, StatusCode: {resp.BaseResp.StatusCode}, StatusMessage: {resp.BaseResp.StatusMessage}",
            )

        if resp.BaseResp.StatusCode != 0:
            raise Exception(
                f"Failed to call gateway service. req_key: {req_key}, StatusCode: {resp.BaseResp.StatusCode}, StatusMessage: {resp.BaseResp.StatusMessage}",
                resp.BaseResp.StatusCode
            )
        return resp

    def process_with_status(self, req_key: str, req_json: str, binary_data: list):
        req = gateway_common_thrift.AlgoReq()
        req.req_key = req_key
        req.binary_data = []
        req.req_json = req_json
        req.binary_data = binary_data
        st = time.time()
        resp = self.gateway_client.Process(req)
        et = time.time()
        mcli.emit_counter("rpc.throughput", 1, tags={"req_key": req_key})
        mcli.emit_timer("rpc.latency", (et - st) * 1000, tags={"req_key": req_key})
        return resp


if __name__ == "__main__":
    REQUEST_KEY = "atomu_aigc_comfyui_ghibli"
    gateway_client = VproxyClient()
    with open("/opt/tiger/vpipe/vpipe/deployment/vgfm_automl/test.jpg", "rb") as f:
        content = f.read()
        rsp = gateway_client.process(REQUEST_KEY, "{}", [content])
    print(rsp)
    binary_data = rsp.binary_data[0]
    import io
    from PIL import Image

    image = Image.open(io.BytesIO(binary_data))
    image.save("result.jpg")
