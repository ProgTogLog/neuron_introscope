from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3_hololens-yolo_mAP-0.82726_epoch-73.pt")
detector.setJsonPath("hololens-yolo_yolov3_detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="holo1.jpg", 
                                            output_image_path="holo1-detected.jpg", 
                                            minimum_percentage_probability = 70
                                            )
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])