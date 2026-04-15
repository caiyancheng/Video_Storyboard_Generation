namespace py IC.AIP
namespace go IC.AIP
namespace cpp IC.AIP


// ErrorCode, ic.aip error code standard
// Doc Ref: https://bytedance.feishu.cn/docs/doccnYJTqGlO4HxcPy2BLuMHNBL

enum ErrorCode {
    Success = 0,


    // /************   Internal Error Defines  ************/
    // /** first three digits start with 100 **/
    // /*
    //  *  the last three digits start with 1 to indicate the back-end main service error
    //  *  the last three digits start with 2 to indicate the algorithm process error
    //  *  the last three digits start with 3 to indicate the company's basic services that depended on have failed
    //  *  the last three digits start with 4 to indicate the other iccv services that depended on have failed
    //  */

    // InternalMainServiceError, only used in some tiny services to indicate all error occured
    InternalMainServiceError = 100000,

    // FileIOError, file io related error, like file read fail, file write fail, etc
    FileIOError = 100101,

    // NetIOError, net io related error, like do http request fail, socket connect fail, etc
    NetIOError = 100102,

    // UnhandledError, unknowned exception that catched in code
    UnhandledError = 100199,


    // AlgorithmConstructorError, construct algorithm net fail
    AlgorithmConstructorError = 100201,

    // AlgorithmPredictError, algorithm net predict fail
    AlgorithmPredictError = 100202,


    // TOSServiceError, tos service related error
    TOSServiceError = 100301,

    // RDSServiceError, rds service related error
    RDSServiceError = 100302,

    // TCCServiceError, tcc service related error
    TCCServiceError = 100303,

    // RedisServiceError, redis service related error
    RedisServiceError = 100304,


    // DependICCVServiceConnectionError, depend iccv services connection fail
    // including services discovery not found instance, instance connect fail, instance connection time out
    DependICCVServiceConnectionError = 100401,

    // DependICCVServiceUnexceptedReturn, dependend services returned abnormal data when all input is normal
    DependICCVServiceUnexceptedReturn = 100402,



    // /************   User Input Error Defines  ************/
    // /** common user input error defines **/
    // /** first three digits start with 201 **/
    // /*
    //  *  the last three digits start with 1 to indicate user auth related error
    //  *  the last three digits start with 2 to indicate common input error
    //  *  the last three digits start with 3 to indicate image input error
    //  *  the last three digits start with 4 to indicate video input error
    //  *  the last three digits start with 5 to indicate prompt text input error
    //  *  the last three digits start with 9 to indicate no need process
    //  */

    // UserAuthFail, failed to identify the user, like user token error
    UserAuthFail = 201101,

    // UserNoPermission, success to identify the user, but user has not permission to do this operation
    UserNoPermission = 201102,


    // ParamMissing, some parameters were not passed by the user
    ParamMissing = 201201,

    // InvalidParam, some parameters' format is not correct
    InvalidParam = 201202,


    // InputImageEmpty, input image data is none
    InputImageEmpty = 201301,

    // InputImageDecodeFail, decode input image fail
    InputImageDecodeFail = 201302,

    // InputImageInvalidColorSpace, input image color space is not satisfied
    InputImageInvalidColorSpace = 201303,

    // InputImageInvalid, input image invalid for algorithm to process
    InputImageInvalid = 201304,

    // InputImageInvalidResolution, unsupport image resolution for algorithm to process
    InputImageInvalidResolution = 201305,


    // InputVideoEmpty, input video data is none
    InputVideoEmpty = 201401,

    InputVideoDecodeFail = 201402,


    InputTextInvalid = 201501,


    // input doesn't need to process
    NoNeedAlgoProcess = 201901,


    // /** image understanding type input error defines **/
    // /** first three digits start with 202 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** human body human face type input error defines **/
    // /** first three digits start with 203 **/
    // /*
    //  *  the last three digits start with 1 to indicate human face error
    //  */

    // ImageNoFaceDetect, no human face found in image
    ImageNoFaceDetect = 203101,

    // ImageNoFaceDetect, no adult male found in image
    ImageNoManFound = 203102,

    // ImageNoWomenFound, no adult female found in image
    ImageNoWomenFound = 203103,

    // ImageFaceInvalidAngle, the face angle is too big or small in image
    ImageFaceInvalidAngle = 203104,


    // /** image effects and editing type input error defines **/
    // /** first three digits start with 204 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** video understanding and editing service error defines **/
    // /** first three digits start with 205 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** ar and vr type input error defines **/
    // /** first three digits start with 206 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** Image quality improvement type input error defines **/
    // /** first three digits start with 207 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** cloud editing type input error defines **/
    // /** first three digits start with 208 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** recommendation type input error defines **/
    // /** first three digits start with 209 **/
    // /*
    //  *  the last three digits start with 1 to indicate xxx
    //  */


    // /** digital human type input error defines **/
    // /** first three digits start with 210 **/
    // /*
    //  *  the last three digits start with 0 to indicate ai model input error
    //  **  the last two digits start with 0 to indicate no clothes found
    //  **  the last two digits start with 1 to indicate invalid clothes found
    //  **  the last two digits start with 2 to indicate mismatching input
    //  */

    // NoClothingDetect, no clothing is found in image
    NoClothingDetect = 210001,

    // TooSmallClothing, input clothing pixels are too small
    TooSmallClothing = 210011,

    // MultiClothes, input multiple clothes in a picture
    MultiClothes = 210012,

    // InvalidClothingKeyPoints, the clothing key points reliability is insufficient
    InvalidClothingKeyPoints = 210013,

    // RotatedClothing, angle based on clothing key points is greater than threshold
    RotatedClothing = 210014,

    // FoldedClothing, clothing is floded
    FoldededClothing = 210015,

    // UnsupportedClothingType, clothing type is not supported
    UnsupportedClothingType = 210016,

    // UnMatchedClothingAndMaskSize
    UnMatchedClothingAndMaskSize = 210021,
}
