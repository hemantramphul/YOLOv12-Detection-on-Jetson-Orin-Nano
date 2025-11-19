import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

import numpy as np
import cv2
from ultralytics import YOLO


def create_pipeline():
    pipeline_desc = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink name=sink max-buffers=1 drop=true"
    )
    return pipeline_desc


def main():
    Gst.init(None)

    pipeline_desc = create_pipeline()
    print(cv2.getBuildInformation())
    print("GStreamer pipeline:")
    print(pipeline_desc)

    pipeline = Gst.parse_launch(pipeline_desc)
    sink = pipeline.get_by_name("sink")

    if sink is None:
        print("Could not find appsink 'sink' in pipeline.")
        return

    # Start pipeline
    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        print("Failed to start pipeline")
        return

    # Load YOLO model
    print("Loading YOLOv12 model (yolo12n.pt)...")
    model = YOLO("yolo12n.pt")  # auto-downloads if missing
    device = 0  # CUDA GPU

    print("Starting inference... Press 'q' or ESC to quit.")

    try:
        while True:
            sample = sink.emit("pull-sample")
            if sample is None:
                print("No sample received from appsink")
                continue

            buf = sample.get_buffer()
            caps = sample.get_caps()
            structure = caps.get_structure(0)
            width = structure.get_value("width")
            height = structure.get_value("height")

            # Map buffer to numpy array
            success, map_info = buf.map(Gst.MapFlags.READ)
            if not success:
                print("Failed to map buffer")
                continue

            frame = np.frombuffer(map_info.data, np.uint8)
            frame = frame.reshape((height, width, 3))
            buf.unmap(map_info)

            # Run YOLO only on 'person' class (0 in COCO)
            results = model(
                frame,
                imgsz=640,
                conf=0.5,
                device=device,
                # classes=[0],  # persons only
                classes=None,
                verbose=False,
            )

            annotated = results[0].plot()

            cv2.imshow("YOLOv12 Detection (GStreamer)", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):  # ESC or q
                break

    finally:
        pipeline.set_state(Gst.State.NULL)
        cv2.destroyAllWindows()
        print("Pipeline stopped, exiting.")


if __name__ == "__main__":
    main()
