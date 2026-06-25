import argparse
import time

import cv2
import numpy as np
from picamera2 import Picamera2, CompletedRequest
from picamera2.devices import IMX500
from picamera2.devices.imx500 import NetworkIntrinsics

DEFAULT_MODEL_PATH = "/usr/share/imx500-models/picosam3_bm.rpk"
WINDOW_NAME  = "PicoSAM3 IMX500 Segmentation"
DISPLAY_SIZE = (640, 480)

SENSOR_W = 4056
SENSOR_H = 3040

RECORD_FPS       = 30.0
ROI_UPDATE_INTERVAL = 0.15

imx500     = None
picam2     = None

latest_mask  = None
latest_frame = None
last_request = None

last_roi_update  = 0.0
last_written_time = 0.0

roi_x = 0
roi_y = 0
roi_w = SENSOR_W
roi_h = SENSOR_H

dragging   = False
drag_start = (0, 0)
drag_end   = (0, 0)


def apply_roi(force=False):
    global last_roi_update
    now = time.time()
    if not force and now - last_roi_update < ROI_UPDATE_INTERVAL:
        return
    rx = max(0, min(roi_x, SENSOR_W - roi_w))
    ry = max(0, min(roi_y, SENSOR_H - roi_h))
    rw = max(64, min(roi_w, SENSOR_W))
    rh = max(64, min(roi_h, SENSOR_H))
    imx500.set_inference_roi_abs((rx, ry, rw, rh))
    last_roi_update = now


def mouse_callback(event, x, y, flags, param):
    global dragging, drag_start, drag_end, roi_x, roi_y, roi_w, roi_h

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging   = True
        drag_start = (x, y)
        drag_end   = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and dragging:
        drag_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        drag_end = (x, y)

        x0, y0 = drag_start
        x1, y1 = drag_end
        dx0, dx1 = sorted([x0, x1])
        dy0, dy1 = sorted([y0, y1])

        sx = int(dx0 * SENSOR_W / DISPLAY_SIZE[0])
        sy = int(dy0 * SENSOR_H / DISPLAY_SIZE[1])
        sw = int((dx1 - dx0) * SENSOR_W / DISPLAY_SIZE[0])
        sh = int((dy1 - dy0) * SENSOR_H / DISPLAY_SIZE[1])

        if sw >= 64 and sh >= 64:
            roi_x, roi_y, roi_w, roi_h = sx, sy, sw, sh
            apply_roi(force=True)
            t_prompt = time.monotonic()

import time
last_e2e_time = 0.0
last_callback_ts = 0.0
t_prompt = 0.0
prompt_latency_ms = None

def segmentation_callback(request: CompletedRequest):
    global latest_mask, latest_frame, last_request, last_e2e_time, last_callback_ts
    global t_prompt, prompt_latency_ms

    now = time.monotonic()
    last_e2e_time = now - last_callback_ts if last_callback_ts else 0.0
    last_callback_ts = now

    if last_e2e_time == 0.0:
        print("\nMetadata keys:", list(request.get_metadata().keys()))

    metadata = request.get_metadata()
    outputs = imx500.get_outputs(metadata)

    if outputs is None:
        return

    if t_prompt:
        prompt_latency_ms = (now - t_prompt) * 1000
        t_prompt = 0.0

    # PicoSAM3 outputs a single-channel logit map — threshold at 0
    mask = (outputs[0][0] > 0).astype(np.uint8) * 255

    frame = request.make_array("main")
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    latest_mask  = mask
    latest_frame = cv2.resize(frame, DISPLAY_SIZE)
    last_request = request


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to .rpk model file")
    args = parser.parse_args()

    model_name = args.model.rsplit("/", 1)[-1].replace(".rpk", "")
    imx500 = IMX500(args.model)
    intr   = imx500.network_intrinsics or NetworkIntrinsics()
    intr.task = "segmentation"
    intr.update_with_defaults()
    intr.inference_rate = 90


    picam2 = Picamera2(imx500.camera_num)
    picam2.start(
        picam2.create_preview_configuration(
            controls={"FrameRate": intr.inference_rate},
            buffer_count=8,
        ),
        show_preview=False,
    )
    picam2.pre_callback = segmentation_callback

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(
        f"{model_name}_segmentation.mp4",
        fourcc,
        RECORD_FPS,
        DISPLAY_SIZE,
    )

    apply_roi(force=True)
    last_written_time = time.time()

    try:
        while True:
            if latest_frame is None or latest_mask is None:
                time.sleep(0.002)
                continue
            
            if last_e2e_time > 0:
                latency_str = f"  |  prompt-to-mask: {prompt_latency_ms:.1f} ms" if prompt_latency_ms else ""
                print(f"Frame interval: {last_e2e_time*1000:.1f} ms  ({1/last_e2e_time:.1f} fps){latency_str}", end="\r")

            now = time.time()
            if now - last_written_time < 1.0 / RECORD_FPS:
                continue
            last_written_time = now

            overlay = latest_frame.copy()
            rx, ry, rw, rh = imx500.get_roi_scaled(last_request)

            roi_mask = cv2.resize(latest_mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
            # Blue overlay for segmentation mask
            overlay[ry:ry+rh, rx:rx+rw, 0] = np.clip(
                overlay[ry:ry+rh, rx:rx+rw, 0].astype(np.float32)
                + roi_mask.astype(np.float32) * 0.5,
                0, 255,
            ).astype(np.uint8)

            # Green ROI border
            cv2.rectangle(overlay, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)

            # Yellow drag rectangle
            if dragging:
                cv2.rectangle(overlay, drag_start, drag_end, (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, overlay)
            video_writer.write(overlay)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    finally:
        video_writer.release()
        picam2.stop()
        cv2.destroyAllWindows()
