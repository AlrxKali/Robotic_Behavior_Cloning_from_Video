from src.training.training_pipeline import RobotTrainingPipeline

def train_model():
    """Loads processed demonstrations, converts them to joint angles, and trains the DAML model."""
    pipeline = RobotTrainingPipeline()
    training_data = pipeline.process_demonstrations("demonstrations", "processed_demonstrations")

    if training_data:
        pipeline.train_model(training_data)
        print("\n✅ Training complete!")
    else:
        print("\n⚠️  No training data found!")

if __name__ == "__main__":
    train_model()
