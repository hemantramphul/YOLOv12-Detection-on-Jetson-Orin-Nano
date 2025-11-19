from ultralytics import YOLO


def main():
    # Load the PyTorch YOLOv12 model
    model = YOLO("yolo12n.pt")  # will download if not present

    # Export to TensorRT engine (FP16, static shape 640x640)
    print("Exporting YOLOv12 model to TensorRT engine (yolo12n.engine)...")
    engine_path = model.export(
        format="engine",   # TensorRT
        device=0,          # GPU 0
        half=True,         # FP16
        imgsz=640,         # input size (640x640)
        dynamic=False,     # static shape (faster)
    )

    print(f"Export done, engine saved to: {engine_path}")


if __name__ == "__main__":
    main()
