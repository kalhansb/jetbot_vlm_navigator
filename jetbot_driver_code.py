import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Int32
from jetbot import Robot  # This imports the library from your notebook
import threading

class JetBotDriver(Node):
    def __init__(self):
        super().__init__('jetbot_driver')
        
        # Initialize the JetBot Robot class
        try:
            self.robot = Robot()
            self.get_logger().info('JetBot Robot instance created successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize JetBot library: {e}')
            return

        # Subscribe to command integers from VLM navigator
        self.subscription = self.create_subscription(
            Int32,
            '/jetbot/cmd',
            self.command_callback,
            10
        )
        
        # Define fixed speeds and rotation parameters
        self.forward_speed = 0.15
        self.turn_speed = 0.15
        
        # Duration for each command before auto-stop (in seconds)
        self.command_duration = 0.2  # Run each command for 200ms then stop
        self.search_duration = 0.1   # Search rotation for 100ms
        
        # Timer to track auto-stop
        self.stop_timer = None
        
        self.get_logger().info('JetBot Driver ready. Command mapping:')
        self.get_logger().info('  1 = FORWARD (auto-stop after 0.2s)')
        self.get_logger().info('  2 = BACKWARD (auto-stop after 0.2s)')
        self.get_logger().info('  3 = TURN_LEFT (auto-stop after 0.2s)')
        self.get_logger().info('  4 = TURN_RIGHT (auto-stop after 0.2s)')
        self.get_logger().info('  5 = SEARCH (auto-stop after 0.1s)')
        self.get_logger().info('  0 = STOP (immediate)')

    def command_callback(self, msg):
        command = msg.data
        
        # Cancel any existing auto-stop timer
        if self.stop_timer is not None:
            self.stop_timer.cancel()
        
        if command == 0:
            # STOP - immediate, no timer
            self.robot.stop()
            self.get_logger().debug('Command: STOP')
            
        elif command == 1:
            # FORWARD - auto-stop after duration
            self.robot.set_motors(self.forward_speed, self.forward_speed)
            self.get_logger().debug('Command: FORWARD')
            self.stop_timer = threading.Timer(self.command_duration, self._auto_stop)
            self.stop_timer.start()
            
        elif command == 2:
            # BACKWARD - auto-stop after duration
            self.robot.set_motors(-self.forward_speed, -self.forward_speed)
            self.get_logger().debug('Command: BACKWARD')
            self.stop_timer = threading.Timer(self.command_duration, self._auto_stop)
            self.stop_timer.start()
            
        elif command == 3:
            # TURN LEFT - auto-stop after duration
            self.robot.set_motors(-self.turn_speed, self.turn_speed)
            self.get_logger().debug('Command: TURN_LEFT')
            self.stop_timer = threading.Timer(self.command_duration, self._auto_stop)
            self.stop_timer.start()
            
        elif command == 4:
            # TURN RIGHT - auto-stop after duration
            self.robot.set_motors(self.turn_speed, -self.turn_speed)
            self.get_logger().debug('Command: TURN_RIGHT')
            self.stop_timer = threading.Timer(self.command_duration, self._auto_stop)
            self.stop_timer.start()
            
        elif command == 5:
            # SEARCH - brief rotation, auto-stop after shorter duration
            self.robot.set_motors(self.turn_speed * 0.5, -self.turn_speed * 0.5)
            self.get_logger().debug('Command: SEARCH (micro-rotation)')
            self.stop_timer = threading.Timer(self.search_duration, self._auto_stop)
            self.stop_timer.start()
            
        else:
            self.get_logger().warn(f'Unknown command: {command}')
            self.robot.stop()
    
    def _auto_stop(self):
        """Called by timer to automatically stop the robot"""
        self.robot.stop()
        self.get_logger().debug('Auto-stopped after command duration')

    def stop_robot(self):
        # Cancel any pending timer
        if self.stop_timer is not None:
            self.stop_timer.cancel()
        # Safety stop
        self.robot.stop()

def main(args=None):
    rclpy.init(args=args)
    node = JetBotDriver()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()