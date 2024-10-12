from vis import *
from trainer import *

def main():


    trainer = YOLO_trainer(
        train_path=r'det/data/train',
        val_path=r'det/data/train',
        test_path=r'det/data/test',
        yaml_path = r'det/data/data.yaml',
        model_size=r'm',
        project_name=r'helmet_detection'
    )
    
    trainer.train(epochs=100, imgsz=640, batch_size=16)
    metrics = trainer.validate()
    exported_model = trainer.export_model(format='onnx')
    results = trainer.predict(r'/home/darky/Documents/f1-inferences/det/data/test.jpg')

if __name__ == "__main__":
    main()