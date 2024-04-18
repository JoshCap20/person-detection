import collections
import torch
import cv2
import time
import os
import logging
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor
from threading import Thread, Lock


# Configuration class
class Config:
    MODEL_NAME = "facebook/detr-resnet-50"
    MODEL_REVISION = "no_timm"
    THRESHOLD = 0.9  # Confidence detection threshold
    FRAME_RATE_CONTROL = 1  # Initial frame rate control (FPS = 1 / FRAME_RATE_CONTROL)
    SAVE_PATH = "detected_images"
    PERSON_LABEL = "person"
    MAX_QUEUE_SIZE = 10  # Maximum size of frame queue to prevent memory overflow


# Logger setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SingletonMeta(type):
    """
    Thread-safe implementation of Singleton.
    """

    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]


class DetectorFactory:
    """
    Factory class for creating object detection models and processors.
    """

    @staticmethod
    def create():
        """
        Creates an object detection model and processor.

        Returns:
            A tuple containing the created model and processor.
        """
        model = DetrForObjectDetection.from_pretrained(
            Config.MODEL_NAME, revision=Config.MODEL_REVISION
        )
        processor = DetrImageProcessor.from_pretrained(
            Config.MODEL_NAME, revision=Config.MODEL_REVISION
        )
        return model, processor


class ActionManager(metaclass=SingletonMeta):
    """
    The ActionManager class handles actions related to object detection.
    Extend this class to add additional actions on person detction.

    Attributes:
        None

    Methods:
        handle_detection(frame, detected_objects): Handles the detection of objects in a frame.
        _save_image(frame, filename): Saves the image frame with a given filename.

    """
    @staticmethod
    def handle_detection(frame, detected_objects):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        ActionManager._save_image(frame, f"person_detected_{timestamp}.jpg")
        logger.info(f"Image saved for detected person: {timestamp}")

    @staticmethod
    def _save_image(frame, filename):
        if not os.path.exists(Config.SAVE_PATH):
            os.makedirs(Config.SAVE_PATH)
        cv2.imwrite(os.path.join(Config.SAVE_PATH, filename), frame)


class ObjectDetector:
    """
    Class for performing object detection using the DETR model.
    """

    model: DetrForObjectDetection
    processor: DetrImageProcessor
    device: torch.device

    def __init__(self):
        """
        Initializes the ObjectDetector class by loading the model and processor.
        """
        logger.info("Loading model and processor")
        self.model, self.processor = DetectorFactory.create()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def process_frame(self, frame):
        """
        Processes a frame for object detection.

        Args:
            frame: The input frame to be processed.

        Returns:
            The detected objects in the frame.
        """
        if not isinstance(frame, Image.Image):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            frame = Image.fromarray(frame)
        
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([frame.size[::-1]], device=self.device)
        return self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=Config.THRESHOLD
        )[0]


class ObjectDetectionApp:
    """
    Application class to run the detection.

    This class represents an application for running object detection on video frames.
    It initializes the necessary components, captures video frames, and processes them
    using an object detector. Detected objects are then passed to an action manager for
    further handling.

    Attributes:
        detector (ObjectDetector): An instance of the object detector.
        action_manager (ActionManager): An instance of the action manager.
        frame_queue (collections.deque): A deque to store video frames.
        running (bool): A flag indicating whether the application is running.

    Methods:
        frame_capture_thread: A thread function for capturing video frames.
        run: The main function to run the object detection application.
    """

    def __init__(self):
        self.detector = ObjectDetector()
        self.action_manager = ActionManager()
        self.frame_queue = collections.deque(maxlen=Config.MAX_QUEUE_SIZE)
        self.running = True

    def frame_capture_thread(self, cap):
        """
        Thread function for capturing video frames.

        Args:
            cap (cv2.VideoCapture): The video capture object.

        This function continuously captures video frames from the specified video capture
        object and appends them to the frame queue. It sleeps for a certain duration to
        control the frame capture rate.
        """
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture video frame - retrying...")
                continue
            self.frame_queue.append(frame)
            time.sleep(max(1 / Config.MAX_QUEUE_SIZE, 0.01))

    def run(self):
        """
        The main function to run the object detection application.

        This function initializes the video capture object, starts the frame capture
        thread, and continuously processes video frames from the frame queue. Detected
        objects are passed to the action manager for handling. The application can be
        terminated by pressing the 'q' key.
        """
        cap = cv2.VideoCapture(0)
        capture_thread = Thread(target=self.frame_capture_thread, args=(cap,))
        capture_thread.start()

        try:
            while self.running:
                if self.frame_queue:
                    frame = self.frame_queue.popleft()
                    results = self.detector.process_frame(frame)
                    if results["scores"].numel() > 0:
                        detected_objects = [
                            self.detector.model.config.id2label[label.item()]
                            for score, label in zip(
                                results["scores"], results["labels"]
                            )
                            if score > Config.THRESHOLD
                            and self.detector.model.config.id2label[label.item()]
                            == Config.PERSON_LABEL
                        ]
                        if detected_objects:
                            logger.info("; ".join(detected_objects))
                            self.action_manager.handle_detection(
                                frame, detected_objects
                            )
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            self.running = False
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Released video capture and destroyed all windows")
            capture_thread.join()


if __name__ == "__main__":
    app = ObjectDetectionApp()
    app.run()
