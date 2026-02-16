#!/usr/bin/env python

# Copyright (c) 2019, The Personal Robotics Lab, The MuSHR Team, The Contributors of MuSHR
# License: BSD 3-Clause. See LICENSE.md file in root directory.

import queue
from threading import Lock
import time
import numpy as np
from rclpy.node import Node
import rclpy
import tf2_ros as tf
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseArray, PoseStamped, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from sensor_msgs.msg import LaserScan
from tf_transformations import euler_from_quaternion, quaternion_from_euler

import mushr_pf.utils as utils
from mushr_pf.motion_model import KinematicMotionModel
from mushr_pf.resample import ReSampler
from mushr_pf.sensor_model import SensorModel

MAP_TOPIC = "/map"


class ParticleFilter(Node):
    """
    Implements particle filtering for estimating the state of the robot car
    """

    def __init__(self):
        """
        Initializes the particle filter
        """
        super().__init__("particle_filter")
        
        # Declare parameters
        self.declare_parameter('car_name', "car")
        self.declare_parameter('publish_tf', False)
        self.declare_parameter('n_particles', 1000)
        self.declare_parameter('n_viz_particles', 60)
        self.declare_parameter('odometry_topic', '/vesc/odom')
        self.declare_parameter('motor_state_topic', '/vesc/sensors/core')
        self.declare_parameter('servo_state_topic', '/vesc/sensors/servo_position_command')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('laser_ray_step', 18)
        self.declare_parameter('exclude_max_range_rays', True)
        self.declare_parameter('max_range_meters', 11.0)
        self.declare_parameter('speed_to_erpm_offset', 0.0)
        self.declare_parameter('speed_to_erpm_gain', 4350)
        self.declare_parameter('steering_angle_to_servo_offset', 0.5)
        self.declare_parameter('steering_angle_to_servo_gain', -1.2135)
        self.declare_parameter('car_length', 0.33)
        
        # Get parameters
        car_name = self.get_parameter('car_name').value
        publish_tf = self.get_parameter('publish_tf').value
        n_particles = self.get_parameter('n_particles').value
        n_viz_particles = self.get_parameter('n_viz_particles').value
        odometry_topic = self.get_parameter('odometry_topic').value
        motor_state_topic = self.get_parameter('motor_state_topic').value
        servo_state_topic = self.get_parameter('servo_state_topic').value
        scan_topic = self.get_parameter('scan_topic').value
        laser_ray_step = self.get_parameter('laser_ray_step').value
        exclude_max_range_rays = self.get_parameter('exclude_max_range_rays').value
        max_range_meters = self.get_parameter('max_range_meters').value
        speed_to_erpm_offset = self.get_parameter('speed_to_erpm_offset').value
        speed_to_erpm_gain = self.get_parameter('speed_to_erpm_gain').value
        steering_angle_to_servo_offset = self.get_parameter('steering_angle_to_servo_offset').value
        steering_angle_to_servo_gain = self.get_parameter('steering_angle_to_servo_gain').value
        car_length = self.get_parameter('car_length').value
        
        self.car_length = car_length  # Store as instance variable for use in methods
        self.PUBLISH_TF = publish_tf
        # The number of particles in this implementation, the total number of particles is constant.
        self.N_PARTICLES = n_particles
        self.N_VIZ_PARTICLES = n_viz_particles  # The number of particles to visualize

        # Cached list of particle indices
        self.particle_indices = np.arange(self.N_PARTICLES)
        # Numpy matrix of dimension N_PARTICLES x 3
        self.particles = np.zeros((self.N_PARTICLES, 3))
        # Numpy matrix containing weight for each particle
        self.weights = np.ones(self.N_PARTICLES) / float(self.N_PARTICLES)

        # Name of car
        self.name = car_name

        # A lock used to prevent concurrency issues. You do not need to worry about this
        self.state_lock = Lock()

        self.tf_buffer = tf.Buffer()

        self.tfl = tf.TransformListener(self.tf_buffer, self)  # Transforms points between coordinate frames

        self.map_info = None

        self.map_service_name = '/map_server/map'

        self.map_client = self.create_client(GetMap, self.map_service_name)

        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Map Server not available, waiting again...')

        def get_map():
            req = GetMap.Request()
            future = self.map_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)

            if future.result() is not None:
                map_msg = future.result().map  # full nav_msgs/OccupancyGrid

                array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))
                permissible_region = np.zeros_like(array_255, dtype=bool)
                permissible_region[array_255 == 0] = 1

                return permissible_region, map_msg.info, map_msg
            else:
                self.get_logger().error('Service call failed %r' % (future.exception(),))
                return None, None
                        
        # Get the map
        self.permissible_region, self.map_info, self.raw_map_msg = get_map()

        # Publish particle filter state
        # Used to create a tf between the map and the laser for visualization
        self.pub_tf = tf.TransformBroadcaster(self)

        # Publishes the expected pose
        self.pose_pub = self.create_publisher(PoseStamped, "inferred_pose", qos_profile=1)
        # Publishes a subsample of the particles
        self.particle_pub = self.create_publisher(PoseArray, "particles", qos_profile=1)
        # Publishes the most recent laser scan
        self.pub_laser = self.create_publisher(LaserScan, "scan", qos_profile=1)
        # Publishes the path of the car
        self.pub_odom = self.create_publisher(Odometry, "odom", qos_profile=1)


        time.sleep(1.0)
        self.initialize_global()

        # An object used for resampling
        self.resampler = ReSampler(self.particles, self.weights, self.state_lock)

        # An object used for applying sensor model
        self.sensor_model = SensorModel(self,
            scan_topic,
            laser_ray_step,
            exclude_max_range_rays,
            max_range_meters,
            self.raw_map_msg,
            self.particles,
            self.weights,
            car_length,
            self.state_lock,
        )

        # An object used for applying kinematic motion model
        self.motion_model = KinematicMotionModel(
            self,
            motor_state_topic,
            servo_state_topic,
            speed_to_erpm_offset,
            speed_to_erpm_gain,
            steering_angle_to_servo_offset,
            steering_angle_to_servo_gain,
            car_length,
            self.particles,
            self.state_lock,
        )

        self.permissible_x, self.permissible_y = np.where(self.permissible_region == 1)

        # Parameters/flags/vars for global localization
        self.global_localize = False
        self.global_suspend = False
        self.ents = None
        self.ents_sum = 0.0
        self.noisy_cnt = 0
        # number of regions to partition. Simulation: 25, Real: 5.
        self.NUM_REGIONS = 25
        # number of updates for regional localization. Simulation 5, Real: 3.
        self.REGIONAL_ROUNDS = 5
        self.regions = []
        self.debug_mode = False
        if self.debug_mode:
            self.global_localize = True  # True when doing global localization

        # precompute regions. Each region is represented by arrays of x, y indices on the map
        region_size = int(len(self.permissible_x) / self.NUM_REGIONS)
        idx = np.argsort(self.permissible_y)  # column-major
        _px, _py = self.permissible_x[idx], self.permissible_y[idx]
        for i in range(self.NUM_REGIONS):
            self.regions.append(
                (
                    _px[i * region_size : (i + 1) * region_size],
                    _py[i * region_size : (i + 1) * region_size],
                )
            )

        # Subscribe to the '/clicked_point' topic. Publised by Foxglove. 
        # See clicked_pose_cb function in this file for more info
        self.click_sub = self.create_subscription(
            PointStamped,
            "/clicked_point",
            self.clicked_point_cb,
            qos_profile=1       )

        # Wait for first laser scan message
        self.last_scan_msg = None
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self._scan_callback, qos_profile=1)
        while self.last_scan_msg is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        
        print("Initialization complete")
    
    def _scan_callback(self, msg):
        """Callback to capture first laser scan"""
        if self.last_scan_msg is None:
            self.last_scan_msg = msg

    def initialize_global(self):
        """
        Initialize the particles to cover the map
        """
        self.state_lock.acquire()
        # Get in-bounds locations
        permissible_x, permissible_y = np.where(self.permissible_region == 1)

        # The number of particles at each location, each with different rotation
        angle_step = 4

        # The sample interval for permissible states
        permissible_step = angle_step * int(len(permissible_x) / self.particles.shape[0])

        # Indices of permissible states to use
        indices = np.arange(0, len(permissible_x), permissible_step)[
            : int((self.particles.shape[0] / angle_step))
        ]

        # Proxy for the new particles
        permissible_states = np.zeros((self.particles.shape[0], 3))

        # Loop through permissible states, each iteration drawing particles with
        # different rotation
        for i in range(angle_step):
            idx_start = i * int(self.particles.shape[0] / angle_step)
            idx_end = (i + 1) * int(self.particles.shape[0] / angle_step)

            permissible_states[idx_start:idx_end, 0] = permissible_y[indices]
            permissible_states[idx_start:idx_end, 1] = permissible_x[indices]
            permissible_states[idx_start:idx_end, 2] = i * (2 * np.pi / angle_step)

        # Transform permissible states to be w.r.t world
        utils.map_to_world(permissible_states, self.map_info)

        # Reset particles and weights
        self.particles[:, :] = permissible_states[:, :]
        self.weights[:] = 1.0 / self.particles.shape[0]
        self.state_lock.release()

    def reinit_cb(self, msg):
        if self.debug_mode:
            self.global_localize = True

    def publish_tf(self, pose, stamp=None):
        """
        Publish a tf between the laser and the map
        This is necessary in order to visualize the laser scan within the map
          pose: The pose of the laser w.r.t the map
          stamp: The time at which this pose was calculated, defaults to None - resulting
                 in using the time at which this function was called as the stamp
        """
        if stamp is None:
            stamp = self.get_clock().now()
        try:
            # Lookup the offset between laser and odom
            delta_off, delta_rot = self.tfl.lookupTransform(
                self.name +"/laser_link", self.name +"/odom", rclpy.time.Time(0)
            )

            # Transform offset to be w.r.t the map
            off_x = delta_off[0] * np.cos(pose[2]) - delta_off[1] * np.sin(pose[2])
            off_y = delta_off[0] * np.sin(pose[2]) + delta_off[1] * np.cos(pose[2])

            # Broadcast the tf
            self.pub_tf.sendTransform(
                (pose[0] + off_x, pose[1] + off_y, 0.0),
                quaternion_from_euler(
                    0, 0, pose[2] + euler_from_quaternion(delta_rot)[2]
                ),
                stamp,
                self.name +"/odom",
                "/map",
            )

        except (tf.LookupException) as e:  # Will occur if odom frame does not exist
            print(e)
            print("failed to find odom")

    def expected_pose(self):
        """
        Uses cosine and sine averaging to more accurately compute average theta
        To get one combined value use the dot product of position and weight vectors
        https://en.wikipedia.org/wiki/Mean_of_circular_quantities

        returns: np array of the expected pose given the current particles and weights
        """
        cosines = np.cos(self.particles[:, 2])
        sines = np.sin(self.particles[:, 2])
        theta = np.arctan2(np.dot(sines, self.weights), np.dot(cosines, self.weights))
        position = np.dot(self.particles[:, 0:2].transpose(), self.weights)
        position[0] += (self.car_length / 2) * np.cos(theta)
        position[1] += (self.car_length / 2) * np.sin(theta)
        return np.array((position[0], position[1], theta), dtype=float)

    def clicked_point_cb(self, msg):
        """
        Reinitialize particles and weights according to the received initial pose
        Applies Gaussian noise to each particle's pose

        msg: PointStamped 
        returns: nothing
        """
        self.state_lock.acquire()
        point = msg.point
        print("SETTING POSE")

        VAR_X = 0.001
        VAR_Y = 0.001
        VAR_THETA = 0.001
        theta = 0.0 
        x = point.x
        y = point.y
        self.particles[:, 0] = np.random.normal(x, VAR_X, self.particles.shape[0])
        self.particles[:, 1] = np.random.normal(y, VAR_Y, self.particles.shape[0])
        self.particles[:, 2] = np.random.normal(
            theta, VAR_THETA, self.particles.shape[0]
        )
        self.weights.fill(1 / self.N_PARTICLES)
        self.state_lock.release()

    def visualize(self):
        """
        Visualize the current state of the filter
           (1) Publishes a tf between the map and the laser. Necessary for visualizing the laser scan in the map
           (2) Publishes the most recent laser measurement. Note that the frame_id of this message should be '/laser'
           (3) Publishes a PoseStamped message indicating the expected pose of the car
           (4) Publishes a subsample of the particles (use self.N_VIZ_PARTICLES).
               Sample so that particles with higher weights are more likely to be sampled.
        """
        self.state_lock.acquire()
        self.inferred_pose = self.expected_pose()

        if isinstance(self.inferred_pose, np.ndarray):
            if self.PUBLISH_TF:
                self.publish_tf(self.inferred_pose)
            ps = PoseStamped()
            ps.header = utils.make_header("map")
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = utils.angle_to_quaternion(self.inferred_pose[2])
            if self.pose_pub.get_num_connections() > 0:
                self.pose_pub.publish(ps)
            if self.pub_odom.get_num_connections() > 0:
                odom = Odometry()
                odom.header = ps.header
                odom.pose.pose = ps.pose
                self.pub_odom.publish(odom)

        if self.particle_pub.get_num_connections() > 0:
            if self.particles.shape[0] > self.N_VIZ_PARTICLES:
                # randomly downsample particles
                proposal_indices = np.random.choice(
                    self.particle_indices, self.N_VIZ_PARTICLES, p=self.weights
                )
                self.publish_particles(self.particles[proposal_indices, :])
            else:
                self.publish_particles(self.particles)

        if self.pub_laser.get_num_connections() > 0 and isinstance(
            self.sensor_model.last_laser, LaserScan
        ):
            self.sensor_model.last_laser.header.frame_id = "/laser"
            self.sensor_model.last_laser.header.stamp = self.get_clock().now().to_msg()
            self.pub_laser.publish(self.sensor_model.last_laser)
        self.state_lock.release()

    def publish_particles(self, particles):
        """
        Helper function for publishing a pose array of particles
          particles: To particles to publish
        """
        pa = PoseArray()
        pa.header = utils.make_header("map")
        pa.poses = utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def set_particles(self, region):
        self.state_lock.acquire()
        # Get in-bounds locations
        permissible_x, permissible_y = region
        assert len(permissible_x) >= self.particles.shape[0]

        # The number of particles at each location, each with different rotation
        angle_step = 4
        # The sample interval for permissible states
        permissible_step = angle_step * len(permissible_x) / self.particles.shape[0]
        # Indices of permissible states to use
        indices = np.arange(0, len(permissible_x), permissible_step)[
            : (self.particles.shape[0] / angle_step)
        ]
        # Proxy for the new particles
        permissible_states = np.zeros((self.particles.shape[0], 3))

        # Loop through permissible states, each iteration drawing particles with
        # different rotation
        for i in range(angle_step):
            idx_start = i * (self.particles.shape[0] / angle_step)
            idx_end = (i + 1) * (self.particles.shape[0] / angle_step)

            permissible_states[idx_start:idx_end, 0] = permissible_y[indices]
            permissible_states[idx_start:idx_end, 1] = permissible_x[indices]
            permissible_states[idx_start:idx_end, 2] = i * (2 * np.pi / angle_step)

        # Transform permissible states to be w.r.t world
        utils.map_to_world(permissible_states, self.map_info)

        # Reset particles and weights
        self.particles[:, :] = permissible_states[:, :]
        self.weights[:] = 1.0 / self.particles.shape[0]
        self.publish_particles(self.particles)
        self.state_lock.release()

    def global_localization(self):
        self.sensor_model.reset_confidence()

        candidate_num = self.particles.shape[0] / self.NUM_REGIONS
        regional_particles = []
        regional_weights = []

        for i in range(self.NUM_REGIONS):
            self.set_particles(self.regions[i])
            cnt_updates = 0
            while cnt_updates < self.REGIONAL_ROUNDS:  # each cluster update
                if self.sensor_model.do_resample:  # if weights updated by sensor model
                    self.resampler.resample_low_variance()
                    self.visualize()
                    self.sensor_model.do_resample = False
                    cnt_updates += 1
            self.state_lock.acquire()
            candidate_indices = np.argsort(self.weights)[-candidate_num:]
            candidates = self.particles[candidate_indices]
            candidates_weights = self.weights[candidate_indices]
            regional_particles.append(candidates.copy())  # save the particles
            regional_weights.append(candidates_weights)  # save the particles' weights
            self.state_lock.release()

        self.state_lock.acquire()
        self.particles[:] = np.concatenate(regional_particles)
        self.weights[:] = np.concatenate(regional_weights)
        self.weights /= self.weights.sum()
        self.global_localize = False
        self.global_suspend = True
        self.sensor_model.do_confidence_update = True
        self.state_lock.release()

    def suspend_update(self):
        self.state_lock.acquire()
        ent = -((self.weights * np.log2(self.weights)).sum())
        print("entropy ==", ent)
        self.ents_sum += ent
        self.ents.put(ent)
        if self.ents.qsize() > 10:
            self.ents_sum -= self.ents.get()
        self.state_lock.release()
        if self.ents_sum / 10 >= 8:
            self.global_suspend = False
        elif 2.0 < ent < 3.0:
            self.resampler.resample_low_variance()


def main(args=None):
    rclpy.init()  # Initialize rclpy
    
    # Create the particle filter node
    pf = ParticleFilter()
    
    # Create a custom executor if needed for the update loop
    try:
        while rclpy.ok():
            # Callbacks are running in separate threads
            
            if pf.sensor_model.confidence < 1e-20 and not pf.global_localize:
                print("=================== KIDNAPPED =====================")
                pf.global_localize = True

            # update particle filter
            if pf.global_localize:  # no resample
                temp = pf.N_VIZ_PARTICLES
                pf.N_VIZ_PARTICLES = 1000
                pf.global_localization()
                pf.visualize()
                pf.N_VIZ_PARTICLES = temp
                pf.ents = queue.Queue()
                pf.ents_sum = 0.0
                pf.noisy_cnt = 0
            # Check if the sensor model says it's time to resample
            elif pf.sensor_model.do_resample:
                # Reset so that we don't keep resampling
                pf.sensor_model.do_resample = False
                pf.resampler.resample_low_variance()
                pf.visualize()  # Perform visualization
            
            # Spin briefly to allow callbacks to execute
            rclpy.spin_once(pf, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pf.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()