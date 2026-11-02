#include <Arduino.h>
#include <driver/i2s.h>
#include "TensorFlowLite_ESP32.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "frontend.h"

// Matches the latest project wiring. Change these if your physical wiring differs.
constexpr int SCK=14,WS=15,SD=13;
alignas(16) static uint8_t arena[96*1024];
static Frontend frontend;
static int8_t ring[FEATURE_ROWS][FEATURE_BINS];
static uint32_t written=0,overflows=0;
static portMUX_TYPE lock=portMUX_INITIALIZER_UNLOCKED;
static tflite::MicroInterpreter* runtime;
static TfLiteTensor* input;
static QueueHandle_t events;
static void fatal(const char* s){Serial.println(s);while(true)delay(1000);}

static void capture(void*){
  int16_t frame[400];int filled=0;int32_t raw[320];
  for(;;){
    size_t bytes=0;
    esp_err_t status=i2s_read(I2S_NUM_0,raw,sizeof(raw),&bytes,pdMS_TO_TICKS(100));
    i2s_event_t event;bool lost=status!=ESP_OK||bytes==0||bytes%4;
    while(xQueueReceive(events,&event,0)==pdTRUE)if(event.type==I2S_EVENT_RX_Q_OVF)lost=true;
    if(lost){filled=0;portENTER_CRITICAL(&lock);written=0;overflows++;portEXIT_CRITICAL(&lock);continue;}
    for(size_t n=0;n<bytes/4;n++){
      frame[filled++]=int16_t(raw[n]>>16);
      if(filled<400)continue;
      int8_t row[FEATURE_BINS];frontend.compute(frame,row,input->params.scale,input->params.zero_point);
      portENTER_CRITICAL(&lock);
      memcpy(ring[written%FEATURE_ROWS],row,FEATURE_BINS);written++;
      portEXIT_CRITICAL(&lock);
      memmove(frame,frame+320,80*sizeof(int16_t));filled=80;
    }
  }
}

void setup(){
  Serial.begin(115200);delay(1000);
  static tflite::MicroErrorReporter reporter;
  static tflite::MicroMutableOpResolver<8> ops;
  ops.AddConv2D();ops.AddDepthwiseConv2D();ops.AddFullyConnected();ops.AddSoftmax();
  ops.AddReshape();ops.AddRelu();ops.AddPad();ops.AddMean();
  const tflite::Model* model=tflite::GetModel(g_model);
  if(model->version()!=TFLITE_SCHEMA_VERSION)fatal("Schema mismatch");
  static tflite::MicroInterpreter interpreter(model,ops,arena,sizeof(arena),&reporter);
  runtime=&interpreter;
  if(runtime->AllocateTensors()!=kTfLiteOk)fatal("Tensor allocation failed: check operators/arena");
  input=runtime->input(0);
  if(input->type!=kTfLiteInt8||input->bytes!=FEATURE_ROWS*FEATURE_BINS)fatal("Input mismatch");
  Serial.printf("TFLM arena used: %u bytes\n",unsigned(runtime->arena_used_bytes()));
  i2s_config_t c={};c.mode=(i2s_mode_t)(I2S_MODE_MASTER|I2S_MODE_RX);c.sample_rate=16000;
  c.bits_per_sample=I2S_BITS_PER_SAMPLE_32BIT;c.channel_format=I2S_CHANNEL_FMT_ONLY_LEFT;
  c.communication_format=I2S_COMM_FORMAT_STAND_I2S;c.dma_buf_count=8;c.dma_buf_len=320;
  i2s_pin_config_t p={};p.mck_io_num=I2S_PIN_NO_CHANGE;p.bck_io_num=SCK;p.ws_io_num=WS;
  p.data_out_num=I2S_PIN_NO_CHANGE;p.data_in_num=SD;
  if(i2s_driver_install(I2S_NUM_0,&c,16,&events)!=ESP_OK||i2s_set_pin(I2S_NUM_0,&p)!=ESP_OK)fatal("I2S init failed");
  Serial.printf("Model %u bytes; reserved arena %u; feature ring %u; free heap %u\n",g_model_len,unsigned(sizeof(arena)),unsigned(sizeof(ring)),ESP.getFreeHeap());
  if(xTaskCreatePinnedToCore(capture,"capture",6144,nullptr,4,nullptr,0)!=pdPASS)fatal("Capture task failed");
  Serial.println("Listening for hello vaani");
}

void loop(){
  static uint32_t lastRun=0,lastDetection=0;static int consecutive=0;
  if(millis()-lastRun<200){delay(5);return;}lastRun=millis();
  bool ready=false;uint32_t lost;
  portENTER_CRITICAL(&lock);
  if(written>=FEATURE_ROWS){
    for(int i=0;i<FEATURE_ROWS;i++)memcpy(input->data.int8+i*FEATURE_BINS,ring[(written-FEATURE_ROWS+i)%FEATURE_ROWS],FEATURE_BINS);
    ready=true;
  }
  lost=overflows;
  portEXIT_CRITICAL(&lock);
  if(!ready){consecutive=0;return;}
  uint32_t start=micros();
  if(runtime->Invoke()!=kTfLiteOk)fatal("Inference failed");
  auto output=runtime->output(0);
  float score=(output->data.int8[0]-output->params.zero_point)*output->params.scale;
  consecutive=score>=DETECTION_THRESHOLD?consecutive+1:0;
  if(consecutive>=2 && (lastDetection==0||millis()-lastDetection>=2000)){
    Serial.printf("HELLO VAANI %.3f\n",score);lastDetection=millis();consecutive=0;
  }
  Serial.printf("score=%.3f inference_ms=%.1f capture_errors=%u\n",score,(micros()-start)/1000.f,lost);
}
