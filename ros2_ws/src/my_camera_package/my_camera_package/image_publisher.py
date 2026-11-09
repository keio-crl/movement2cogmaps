import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ImagePublisher(Node):

    def __init__(self):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, 'image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hzで更新
        self.br = CvBridge()
        self.cap = cv2.VideoCapture(0)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            image_msg = self.br.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(image_msg)
            self.get_logger().info('Publishing Image')

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.cap.release()

if __name__ == '__main__':
    main()
