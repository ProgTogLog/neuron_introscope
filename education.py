from imageai.Detection.Custom import DetectionModelTrainer

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory="Dataset")
trainer.setTrainConfig(object_names_array=["ammo", "firearms", "grenade", "knife"], batch_size=4, num_experiments=200, train_from_pretrained_model="yolov3.pt")
trainer.trainModel()