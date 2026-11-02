# Hello Vaani TinyML — fresh MFCC and MFE training

This project trains two small depthwise CNNs from random initialization for an ESP32-S3-WROOM-1-N16R8. It does not reuse the previous model or its preprocessing. MFE and MFCC are compared as alternative inputs; running both networks on the board is unnecessary. The selected model is chosen on validation results, never test results.

## Dataset and evaluation

Default dataset location: `C:\Users\jayde\OneDrive\Desktop\data set`.
Class order: `hello_vaani`, `noise`, `silence`, `speech`.

- Decode and remove exact audio duplicates before splitting.
- Exclude `keyword_augmented`: its original-recording mapping is unavailable, so safe separation from validation/test cannot be established.
- Split remaining real recordings 70%/15%/15% into training/validation/test with a fixed seed.
- Use `keyword_tts` only for training. Create new shifts, gain changes and noise mixes only from training audio.
- Class-weight the loss to reduce keyword-class dominance.
- Validation selects the checkpoint, INT8 detection threshold and model. Test evaluates the fixed model and threshold.

`manifest.json` records every accepted file, decoded-audio hash and split. `dataset_audit.json` records exclusions and split counts. Speaker/session identities are unavailable: these are recording-disjoint results, **not proof of accuracy on unseen speakers**. Exact duplicate detection does not detect all transformed copies.

Negative examples are mainly one second long. They are repeated to fill the two-second window rather than giving only negatives a distinctive zero-padding pattern. This still differs from live continuous speech. More real microphone recordings, long conversational audio and hard negatives such as “hello”, “vaani”, and similar phrases are needed for a realistic false-alarm benchmark.

## Frontend contract

Both paths use 16 kHz mono audio, 32,000 samples, 400-sample periodic Hann windows, 320-sample hop, FFT512 and 99 complete frames (no centered or final padded frame). Spectrum power is squared FFT magnitude divided by 512. There are 32 triangular HTK mel bands between 80 and 7,600 Hz, with natural logarithm and floor 1e-10.

- **MFE:** all 32 log-mel energies; input `[1,99,32,1]`.
- **MFCC:** orthonormal DCT-II of the same log-mel energies, first 13 coefficients including C0; input `[1,99,13,1]`.

Per-coefficient mean and standard deviation are learned from training data only and exported. No pre-emphasis, peak normalization, or automatic gain control is used. PCM16 is divided by 32768. Model input/output and learned weights use INT8; frontend FFT/log calculations use floating point.

## Reproduce

Use Python 3.11 with the versions in `requirements-lock.txt` (written after training). On this computer the verified training interpreter is `C:\Users\jayde\.platformio\penv\Scripts\python.exe`.

```powershell
python train.py --dataset 'C:\Users\jayde\OneDrive\Desktop\data set' --epochs 40 --cache features.npz
python export_firmware.py
```

Use a new cache path whenever changing dataset contents or preprocessing. A cache is a snapshot and is not automatically invalidated. Training can take several minutes or longer on CPU.

Each feature folder contains the Keras checkpoint, INT8 TFLite model, C byte array, exact frontend constants, training history, quantized validation/test metrics and test vectors. `comparison.json` records model selection. The generated PlatformIO project contains the selected model.

## ESP32-S3 / PlatformIO

Open the `platformio` folder. Board settings select 16 MB flash and OPI PSRAM. Connect an INMP441 with SCK=GPIO14, WS=GPIO15, SD=GPIO13, VDD=3.3V, GND=GND and L/R=GND. These pins match the latest existing firmware; earlier chat examples swapped SCK/WS, so follow the physical connections.

Build with `pio run`, upload with `pio run -t upload`, then open `pio device monitor`. Default serial configuration targets USB-to-UART. For native USB, add `-DARDUINO_USB_MODE=1` and `-DARDUINO_USB_CDC_ON_BOOT=1` to `build_flags`.

The firmware retains only an INT8 feature ring and a 400-sample PCM frame, rather than a full 64 KB PCM recording. Capture runs on a separate task. Overflow resets the feature history to avoid inference across missing audio. The default tensor arena reserves 96 KiB; this is a reservation, not measured minimum consumption. Serial output reports capture errors, inference time and free heap. Two consecutive scores over the validation threshold trigger a detection with a two-second cooldown. These streaming rules are additional to the offline clip metrics and require live validation.

Model-file bytes are flash storage, not total RAM. The reported largest tensor excludes allocator metadata, simultaneous tensors, scratch buffers, audio DMA, task stacks and frontend workspace. Build size also differs from runtime heap usage. Do not assume offline accuracy or desktop latency establishes live ESP32 performance.

## References

- [TensorFlow MFCC documentation](https://www.tensorflow.org/api_docs/python/tf/signal/mfccs_from_log_mel_spectrograms)
- [TensorFlow integer quantization](https://www.tensorflow.org/model_optimization/guide/quantization/post_training)
- [TensorFlow Lite Micro](https://github.com/tensorflow/tflite-micro)
