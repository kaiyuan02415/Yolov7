import time
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO

if __name__ == "__main__":
    # 'predict':单张图片预测,'video':视频检测,'fps':表示测试fps
    mode = "video"
    crop            = False
    count           = False
    video_save_path = r"video_out"
    video_path = r"videos/Traffic1.mp4"
    video_fps       = 25.0
    test_interval   = 100
    fps_image_path  = "img/street.jpg"
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    heatmap_save_path = "model_data/heatmap_vision.png"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    yolo = YOLO()

    # 直接指定video
    if mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            if not ref:
                break
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)
            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            # print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'export_onnx', 'dir_predict'.")
