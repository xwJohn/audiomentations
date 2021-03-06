/*
 *Copyright 2015 Alibaba Group Holding Limited
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 */

#include <pthread.h>
#include <unistd.h>
#include <ctime>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <fstream>
#include "nlsClient.h"
#include "nlsEvent.h"
#include "speechTranscriberRequest.h"
#include "nlsCommonSdk/Token.h"

#define FRAME_SIZE 3200
#define SAMPLE_RATE 16000
using namespace AlibabaNlsCommon;
using AlibabaNls::NlsClient;
using AlibabaNls::NlsEvent;
using AlibabaNls::LogDebug;
using AlibabaNls::LogInfo;
using AlibabaNls::SpeechTranscriberRequest;

// 自定义线程参数
struct ParamStruct {
    std::string fileName;
    std::string token;
    std::string appkey;
};

// 自定义事件回调参数
struct ParamCallBack {
    int userId;
    char userInfo[10];
};

//全局维护一个服务鉴权token和其对应的有效期时间戳，
//每次调用服务之前，首先判断token是否已经过期，
//如果已经过期，则根据AccessKey ID和AccessKey Secret重新生成一个token，并更新这个全局的token和其有效期时间戳。
//注意：不要每次调用服务之前都重新生成新token，只需在token即将过期时重新生成即可。所有的服务并发可共用一个token。
std::string g_akId = "";
std::string g_akSecret = "";
std::string g_token = "";
long g_expireTime = -1;

// 根据AccessKey ID和AccessKey Secret重新生成一个token，并获取其有效期时间戳
// token使用规则：在有效期到期前可以一直使用，且可以多个进程/多个线程/多个应用使用均可，建议在有效期快到期时提起申请新的token
int generateToken(std::string akId, std::string akSecret, std::string* token, long* expireTime) {
    NlsToken nlsTokenRequest;
    nlsTokenRequest.setAccessKeyId(akId);
    nlsTokenRequest.setKeySecret(akSecret);

    if (-1 == nlsTokenRequest.applyNlsToken()) {
        // 获取失败原因
        printf("generateToken Failed: %s\n", nlsTokenRequest.getErrorMsg());
        return -1;
    }

    *token = nlsTokenRequest.getToken();
    *expireTime = nlsTokenRequest.getExpireTime();
    return 0;
}

//@brief 获取sendAudio发送延时时间
//@param dataSize 待发送数据大小
//@param sampleRate 采样率 16k/8K
//@param compressRate 数据压缩率，例如压缩比为10:1的16k opus编码，此时为10；非压缩数据则为1
//@return 返回sendAudio之后需要sleep的时间
//@note 对于8k pcm 编码数据, 16位采样，建议每发送1600字节 sleep 100 ms.
// 对于16k pcm 编码数据, 16位采样，建议每发送3200字节 sleep 100 ms.
// 对于其它编码格式的数据, 用户根据压缩比, 自行估算, 比如压缩比为10:1的 16k opus,
// 需要每发送3200/10=320 sleep 100ms.
unsigned int getSendAudioSleepTime(int dataSize, int sampleRate, int compressRate) {
    // 仅支持16位采样
    const int sampleBytes = 16;
    // 仅支持单通道
    const int soundChannel = 1;

    // 当前采样率，采样位数下每秒采样数据的大小
    int bytes = (sampleRate * sampleBytes * soundChannel) / 8;
    // 当前采样率，采样位数下每毫秒采样数据的大小
    int bytesMs = bytes / 1000;
    // 待发送数据大小除以每毫秒采样数据大小，以获取sleep时间
    int sleepMs = (dataSize * compressRate) / bytesMs;
    return sleepMs;
}

//@brief 调用start(), 成功与云端建立连接, sdk内部线程上报started事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onTranscriptionStarted(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onTranscriptionStarted: %d\n", tmpParam->userId);
    // 当前任务的task id，方便定位问题，建议输出，特别提醒该taskid非常重要，是和服务端交互的唯一标识，因此建议在实际使用时建议输出该taskid
    printf("onTranscriptionStarted: status code=%d, task id=%s\n", cbEvent->getStatusCode(), cbEvent->getTaskId());
    // 获取服务端返回的全部信息
    //printf("onTranscriptionStarted: all response=%s\n", cbEvent->getAllResponse());
}

//@brief 服务端检测到了一句话的开始, sdk内部线程上报SentenceBegin事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onSentenceBegin(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onSentenceBegin: %d\n", tmpParam->userId);
    // 当前任务的task id，方便定位问题，建议输出，特别提醒该taskid非常重要，是和服务端交互的唯一标识，因此建议在实际使用时建议输出该taskid
    printf("onSentenceBegin: status code=%d, task id=%s, index=%d, time=%d\n", cbEvent->getStatusCode(), cbEvent->getTaskId(),
                cbEvent->getSentenceIndex(), //句子编号，从1开始递增
                cbEvent->getSentenceTime() //当前已处理的音频时长，单位是毫秒
                );
    // 获取服务端返回的全部信息
    //printf("onTranscriptionStarted: all response=%s\n", cbEvent->getAllResponse());
}

//@brief 服务端检测到了一句话结束, sdk内部线程上报SentenceEnd事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onSentenceEnd(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onSentenceEnd: %d\n", tmpParam->userId);
    // 当前任务的task id，方便定位问题，建议输出，特别提醒该taskid非常重要，是和服务端交互的唯一标识，因此建议在实际使用时建议输出该taskid
    printf("onSentenceEnd: status code=%d, task id=%s, index=%d, time=%d, begin_time=%d, result=%s\n", cbEvent->getStatusCode(), cbEvent->getTaskId(),
                cbEvent->getSentenceIndex(), //句子编号，从1开始递增
                cbEvent->getSentenceTime(), //当前已处理的音频时长，单位是毫秒
                cbEvent->getSentenceBeginTime(), // 对应的SentenceBegin事件的时间
                cbEvent->getResult()    // 当前句子的完整识别结果
                );
        //  << ", confidence: " << cbEvent->getSentenceConfidence()    // 结果置信度,取值范围[0.0,1.0]，值越大表示置信度越高
        //  << ", stashResult begin_time: " << cbEvent->getStashResultBeginTime() //下一句话开始时间
        //  << ", stashResult current_time: " << cbEvent->getStashResultCurrentTime() //下一句话当前时间
        //  << ", stashResult Sentence_id: " << cbEvent->getStashResultSentenceId() //sentence Id
        //  << ", stashResult Text: " << cbEvent->getStashResultText() //下一句话前缀
    // 获取服务端返回的全部信息
    //printf("onTranscriptionStarted: all response=%s\n", cbEvent->getAllResponse());
}

//@brief 识别结果发生了变化, sdk在接收到云端返回到最新结果时, sdk内部线程上报ResultChanged事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onTranscriptionResultChanged(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onTranscriptionResultChanged: %d\n", tmpParam->userId);
    // 当前任务的task id，方便定位问题，建议输出，特别提醒该taskid非常重要，是和服务端交互的唯一标识，因此建议在实际使用时建议输出该taskid
    printf("onTranscriptionResultChanged: status code=%d, task id=%s, index=%d, time=%d, result=%s\n", cbEvent->getStatusCode(), cbEvent->getTaskId(),
                cbEvent->getSentenceIndex(), //句子编号，从1开始递增
                cbEvent->getSentenceTime(), //当前已处理的音频时长，单位是毫秒
                cbEvent->getResult()    // 当前句子的完整识别结果
                );
    // 获取服务端返回的全部信息
    //printf("onTranscriptionStarted: all response=%s\n", cbEvent->getAllResponse());
}

//@brief 服务端停止实时音频流识别时, sdk内部线程上报Completed事件
//@note 上报Completed事件之后，SDK内部会关闭识别连接通道. 此时调用sendAudio会返回-1, 请停止发送.
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onTranscriptionCompleted(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onTranscriptionCompleted: %d\n", tmpParam->userId);
    printf("onTranscriptionCompleted: status code=%d, task id=%s\n", cbEvent->getStatusCode(), cbEvent->getTaskId());
}

//@brief 识别过程(包含start(), send(), stop())发生异常时, sdk内部线程上报TaskFailed事件
//@note 上报TaskFailed事件之后, SDK内部会关闭识别连接通道. 此时调用sendAudio会返回-1, 请停止发送
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onTaskFailed(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onTaskFailed: %d\n", tmpParam->userId);
    printf("onTaskFailed: status code=%d, task id=%s, error message=%s\n", cbEvent->getStatusCode(), cbEvent->getTaskId(), cbEvent->getErrorMessage());
    // 获取服务端返回的全部信息
    //printf("onTaskFailed: all response=%s\n", cbEvent->getAllResponse());
}

//@brief sdk内部线程上报语音表单结果事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onSentenceSemantics(NlsEvent* cbEvent, void* cbParam) {
    ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    // 演示如何打印/使用用户自定义参数示例
    printf("onSentenceSemantics: %d\n", tmpParam->userId);
    // 获取服务端返回的全部信息
    printf("onSentenceSemantics: all response=%s\n", cbEvent->getAllResponse());
}

//@brief 识别结束或发生异常时，会关闭连接通道, sdk内部线程上报ChannelCloseed事件
//@param cbEvent 回调事件结构, 详见nlsEvent.h
//@param cbParam 回调自定义参数，默认为NULL, 可以根据需求自定义参数
void onChannelClosed(NlsEvent* cbEvent, void* cbParam) {
	ParamCallBack* tmpParam = (ParamCallBack*)cbParam;
    delete tmpParam; //识别流程结束,释放回调参数
}

// 工作线程
void* pthreadFunc(void* arg) {
    int sleepMs = 0;
    ParamCallBack *cbParam = NULL;
    //初始化自定义回调参数, 以下两变量仅作为示例表示参数传递, 在demo中不起任何作用
    //回调参数在堆中分配之后, SDK在销毁requesr对象时会一并销毁, 外界无需在释放
    cbParam = new ParamCallBack;
    cbParam->userId = 1234;
    strcpy(cbParam->userInfo, "User.");

    // 0: 从自定义线程参数中获取token, 配置文件等参数.
    ParamStruct* tst = (ParamStruct*)arg;
    if (tst == NULL) {
        printf("arg is not valid\n");
        return NULL;
    }

    /* 打开音频文件, 获取数据 */
    std::ifstream fs;
    fs.open(tst->fileName.c_str(), std::ios::binary | std::ios::in);
    if (!fs) {
        printf("%s isn't exist..\n", tst->fileName.c_str());
        return NULL;
    }

    //2: 创建实时音频流识别SpeechTranscriberRequest对象
    SpeechTranscriberRequest* request = NlsClient::getInstance()->createTranscriberRequest();
    if (request == NULL) {
        printf("createTranscriberRequest failed.\n");
        return NULL;
    }

    request->setOnTranscriptionStarted(onTranscriptionStarted, cbParam);                // 设置识别启动回调函数
    request->setOnTranscriptionResultChanged(onTranscriptionResultChanged, cbParam);    // 设置识别结果变化回调函数
    request->setOnTranscriptionCompleted(onTranscriptionCompleted, cbParam);            // 设置语音转写结束回调函数
    request->setOnSentenceBegin(onSentenceBegin, cbParam);                              // 设置一句话开始回调函数
    request->setOnSentenceEnd(onSentenceEnd, cbParam);                                  // 设置一句话结束回调函数
    request->setOnTaskFailed(onTaskFailed, cbParam);                                    // 设置异常识别回调函数
    request->setOnChannelClosed(onChannelClosed, cbParam);                              // 设置识别通道关闭回调函数
    request->setOnSentenceSemantics(onSentenceSemantics, cbParam);                      //设置二次结果返回回调函数, 开启enable_nlp后返回

    request->setAppKey(tst->appkey.c_str());            // 设置AppKey, 必填参数, 请参照官网申请
	request->setFormat("pcm");                          // 设置音频数据编码格式, 默认是pcm
	request->setSampleRate(SAMPLE_RATE);                // 设置音频数据采样率, 可选参数，目前支持16000, 8000. 默认是16000
	request->setIntermediateResult(true);               // 设置是否返回中间识别结果, 可选参数. 默认false
	request->setPunctuationPrediction(true);            // 设置是否在后处理中添加标点, 可选参数. 默认false
	request->setInverseTextNormalization(true);         // 设置是否在后处理中执行数字转写, 可选参数. 默认false

    //语音断句检测阈值，一句话之后静音长度超过该值，即本句结束，合法参数范围200～2000(ms)，默认值800ms
    //request->setMaxSentenceSilence(800);
    //request->setCustomizationId("TestId_123"); //定制模型id, 可选.
    //request->setVocabularyId("TestId_456"); //定制泛热词id, 可选.
    // 用于传递某些定制化、高级参数设置，参数格式为json格式： {"key": "value"}
    //request->setPayloadParam("{\"vad_model\": \"farfield\"}");
    //设置是否开启词模式
    request->setPayloadParam("{\"enable_words\": true}");

    //语义断句， 默认false，非必需则不建议设置
    //request->setPayloadParam("{\"enable_semantic_sentence_detection\": false}");
    //是否开启顺滑，默认不开，非必需则不建议设置
    //request->setPayloadParam("{\"disfluency\": true}");

    //设置vad的模型，默认不设置，非必需则不建议设置
    //request->setPayloadParam("{\"vad_model\": \"farfield\"}");
    //设置是否忽略单句超时
    //request->setPayloadParam("{\"enable_ignore_sentence_timeout\": false}");
    //vad断句开启后处理，默认不设置，非必需则不建议设置
    //request->setPayloadParam("{\"enable_vad_unify_post\": true}");

    request->setToken(tst->token.c_str());

    //3: start()为异步操作。成功返回started事件。失败返回TaskFailed事件。
    if (request->start() < 0) {
		printf("start() failed. may be can not connect server. please check network or firewalld\n");
        NlsClient::getInstance()->releaseTranscriberRequest(request); // start()失败，释放request对象
        return NULL;
    }

    while (!fs.eof()) {
        uint8_t data[FRAME_SIZE] = {0};

        fs.read((char *)data, sizeof(uint8_t) * FRAME_SIZE);
        size_t nlen = fs.gcount();
        if (nlen <= 0) {
            continue;
        }

        //4: 发送音频数据. sendAudio返回-1表示发送失败, 需要停止发送.
        int ret = request->sendAudio(data, nlen);
        if (ret < 0) {
            // 发送失败, 退出循环数据发送
            printf("send data fail.\n");
            break;
        }

        //语音数据发送控制：
        //语音数据是实时的, 不用sleep控制速率, 直接发送即可.
        //语音数据来自文件, 发送时需要控制速率, 使单位时间内发送的数据大小接近单位时间原始语音数据存储的大小.
        sleepMs = getSendAudioSleepTime(nlen, SAMPLE_RATE, 1); // 根据 发送数据大小，采样率，数据压缩比 来获取sleep时间

        //5: 语音数据发送延时控制
        usleep(sleepMs * 1000);
    }

    // 关闭音频文件
    fs.close();

    //6: 通知云端数据发送结束.
    //stop()为异步操作.失败返回TaskFailed事件。
    request->stop();
    //7: 识别结束, 释放request对象
    NlsClient::getInstance()->releaseTranscriberRequest(request);
    return NULL;
}

//识别单个音频数据
int speechTranscriberFile(const char* appkey) {
    // 获取当前系统时间戳，判断token是否过期
    std::time_t curTime = std::time(0);
    if (g_expireTime - curTime < 10) {
		printf("the token will be expired, please generate new token by AccessKey-ID and AccessKey-Secret.\n");
        if (-1 == generateToken(g_akId, g_akSecret, &g_token, &g_expireTime)) {
            return -1;
        }
    }

    ParamStruct pa;
    pa.token = g_token;
    pa.appkey = appkey;
    pa.fileName = "test0.wav";

    pthread_t pthreadId;
    // 启动一个工作线程, 用于识别
    pthread_create(&pthreadId, NULL, &pthreadFunc, (void *)&pa);
    pthread_join(pthreadId, NULL);
	return 0;
}

//识别多个音频数据;
//sdk多线程指一个音频数据对应一个线程, 非一个音频数据对应多个线程.
//示例代码为同时开启2个线程识别2个文件;
//免费用户并发连接不能超过2个;
#define AUDIO_FILE_NUMS 2
#define AUDIO_FILE_NAME_LENGTH 32
int speechTranscriberMultFile(const char* appkey) {
    // 获取当前系统时间戳，判断token是否过期
    std::time_t curTime = std::time(0);
    if (g_expireTime - curTime < 10) {
		printf("the token will be expired, please generate new token by AccessKey-ID and AccessKey-Secret.\n");
        if (-1 == generateToken(g_akId, g_akSecret, &g_token, &g_expireTime)) {
            return -1;
        }
    }

    char audioFileNames[AUDIO_FILE_NUMS][AUDIO_FILE_NAME_LENGTH] = {"test0.wav", "test1.wav"};
    ParamStruct pa[AUDIO_FILE_NUMS];
    for (int i = 0; i < AUDIO_FILE_NUMS; i ++) {
        pa[i].token = g_token;
        pa[i].appkey = appkey;
        pa[i].fileName = audioFileNames[i];
    }

    std::vector<pthread_t> pthreadId(AUDIO_FILE_NUMS);
    // 启动2个工作线程, 同时识别2个音频文件
    for (int j = 0; j < AUDIO_FILE_NUMS; j++) {
        pthread_create(&pthreadId[j], NULL, &pthreadFunc, (void *)&(pa[j]));
    }
    for (int j = 0; j < AUDIO_FILE_NUMS; j++) {
        pthread_join(pthreadId[j], NULL);
    }
	return 0;
}

int main(int arc, char* argv[]) {
    if (arc < 4) {
		printf("params is not valid. Usage: ./demo <your appkey> <your AccessKey ID> <your AccessKey Secret>\n");
        return -1;
    }

    std::string appkey = argv[1];
    g_akId = argv[2];
    g_akSecret = argv[3];

    // 根据需要设置SDK输出日志, 可选. 此处表示SDK日志输出至log-Transcriber.txt， LogDebug表示输出所有级别日志
    int ret = NlsClient::getInstance()->setLogConfig("log-transcriber", LogDebug);
    if (-1 == ret) {
		printf("set log failed\n");
        return -1;
    }

    //启动工作线程
    NlsClient::getInstance()->startWorkThread(4);

    // 识别单个音频数据
    speechTranscriberFile(appkey.c_str());

    // 识别多个音频数据
    // speechTranscriberMultFile(appkey.c_str());

    // 所有工作完成，进程退出前，释放nlsClient. 请注意, releaseInstance()非线程安全.
    NlsClient::releaseInstance();
    return 0;
}
