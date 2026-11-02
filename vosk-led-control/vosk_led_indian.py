"""Receive live PCM audio from ESP32 and print Vosk speech recognition."""
import argparse
import json
import sys
import time

import serial
from vosk import KaldiRecognizer, Model, SetLogLevel


COLOR_COMMANDS = {
    "red": b"LED RED\n",
    "green": b"LED GREEN\n",
    "blue": b"LED BLUE\n",
    "white": b"LED WHITE\n",
    "yellow": b"LED YELLOW\n",
    "purple": b"LED PURPLE\n",
    "violet": b"LED PURPLE\n",
    "cyan": b"LED CYAN\n",
    "orange": b"LED ORANGE\n",
}


def led_command(text):
    words = set(text.lower().split())
    if "off" in words and ({"light", "led"} & words):
        return b"LED OFF\n", "off"
    for color, command in COLOR_COMMANDS.items():
        if color in words and ({"light", "led"} & words):
            shown_color = "purple" if color == "violet" else color
            return command, shown_color
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="COM6")
    parser.add_argument("--model", default="model", help="Folder containing a Vosk model")
    args = parser.parse_args()

    SetLogLevel(-1)
    model = Model(args.model)
    # The full Indian-English model does not support runtime grammar graphs.
    # Use normal recognition and filter the final text for LED commands.
    recognizer = KaldiRecognizer(model, 16000)

    with serial.Serial(args.port, 460800, timeout=1, write_timeout=2) as device:
        device.dtr = False
        device.rts = False
        # Opening the USB-UART port can reset the ESP32. Let it boot before
        # sending commands, then retry START until its handshake arrives.
        time.sleep(2)
        device.reset_input_buffer()
        ready = False
        for _ in range(5):
            device.write(b"START\n")
            deadline = time.monotonic() + 1
            while time.monotonic() < deadline:
                line = device.readline().strip()
                if line == b"ASR_READY 16000 PCM16":
                    ready = True
                    break
                if line.startswith(b"ERROR"):
                    raise RuntimeError(line.decode(errors="replace"))
            if ready:
                break
        if not ready:
            raise RuntimeError("ESP32 recorder did not respond. Upload the asr-recorder environment.")

        time.sleep(3)
        print("HELLO VAANI DETECTED", flush=True)
        print("Listening. Press Ctrl+C to stop.", flush=True)
        try:
            while True:
                # Smaller chunks reduce serial buffering latency for short commands.
                audio = device.read(1024)
                if not audio:
                    continue
                if recognizer.AcceptWaveform(audio):
                    result = json.loads(recognizer.Result()).get("text", "")
                    if result:
                        print(result, flush=True)
                        command, color = led_command(result)
                        if command:
                            device.write(command)
                            print(f"LED -> {color}", flush=True)
        except KeyboardInterrupt:
            print()
        finally:
            device.write(b"STOP\n")
            final = json.loads(recognizer.FinalResult()).get("text", "")
            if final:
                print(final)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        raise SystemExit(f"ASR failed: {error}")
