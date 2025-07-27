import numpy as np
import random
import pygame
import sys
import math
import matplotlib.pyplot as plt
from math import sqrt
from KalmanFilter import KalmanFilter
import cv2

# Initialize Pygame
pygame.init()

# Set up display
width, height = 150, 1000
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Car Game")

# Set up the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('pygame_video.avi', fourcc, 30.0, (width, height))

# Colors
BLUE = (135, 206, 235)  # Sky blue background
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
ORANGE = (255, 165, 0)

# Font setup
font = pygame.font.Font(None, 18)

# Car settings
car_width, car_height = 15, 30
car1_image = pygame.image.load('car_red.jpg').convert_alpha()
car1_image = pygame.transform.scale(car1_image, (car_width, car_height))
car2_image = pygame.image.load('car_black.jpg').convert_alpha()
car2_image = pygame.transform.scale(car2_image, (car_width, car_height))

pedestrian_width, pedestrian_height = 10, 10
pedestrian_image = pygame.image.load('pedestrian.png').convert_alpha()
pedestrian_image = pygame.transform.scale(pedestrian_image, (pedestrian_width, pedestrian_height))

rel_object_width, rel_object_height = 10, 10
rel_object_image = pygame.image.load('tree.jpg').convert_alpha()
rel_object_image = pygame.transform.scale(rel_object_image, (rel_object_width, rel_object_height))

car1_x, car1_y = 7 * width // 12 - car_width // 2, height - car_height - 100
car2_x, car2_y = 5 * width // 12 - car_width // 2, height - car_height - 400
pedestrian_x, pedestrian_y = 1 * width // 12 - pedestrian_width // 2, 100
rel_object_x, rel_object_y = 0, 800

# Road settings
line_width = 3
line_height = 20
line_gap = 10
line_x1 = 2 * width // 6
line_x2 = 3 * width // 6
line_x3 = 4 * width // 6
line_x4 = 5 * width // 6

line_y = 0
speed_car_1 = -10
speed_car_2 = -12
pedestrian_speed_x = 1
pedestrian_speed_y = 0.1
rel_object_speed = 0

real_positions_car_1 = []
estimations_car_1 = []

car1_trace = []

real_positions_car_2 = []
estimations_car_2 = []

squared_diffs_car_1 = []
squared_diffs_car_2 = []

transformed_prediction_to_car_1 = []
uncertainty_values = []

dt = 1.0 / 60  # Assuming 60 FPS
std_acc = 0.3  # Example process noise
x_std_meas, y_std_meas = 0.3, 0.3  # Example measurement noise
kf2 = KalmanFilter(dt, 0, 0, std_acc, x_std_meas, y_std_meas)
kf_1_wrt_2 = KalmanFilter(dt, 0, 0, std_acc, x_std_meas, y_std_meas)
kf_ped_wrt_car_2 = KalmanFilter(dt, 0, 0, std_acc, x_std_meas, y_std_meas)

small_time_step = 1

def draw_halo(surface, position, radius, color, alpha):
    # Create a new surface with per-pixel alpha
    halo_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
    pygame.draw.circle(halo_surface, color, (radius, radius), radius)

    # Set the transparency level
    halo_surface.set_alpha(alpha)

    # Blit this surface onto the main surface
    surface.blit(halo_surface, (position[0] - radius, position[1] - radius))

def attractive_force(current_pos, obstacle_pos, strength=200.0, effective_range=200):
    # Simple linear attractive force towards the goal
    distance = np.array(current_pos) - np.array(obstacle_pos)
    distance_norm = np.linalg.norm(distance)

    if distance_norm < effective_range and np.sign(current_pos[0] - obstacle_pos[0]) > 0:
        return -strength * (1 / distance_norm - 1 / effective_range)
    else:
        return np.array([0])

def repulsive_force_car(current_pos, obstacle_pos, strength=150.0, effective_range=50):
    # Repulsive force away from the obstacle
    distance = np.array(current_pos) - np.array(obstacle_pos)
    distance_norm = np.linalg.norm(distance)
    if distance_norm < effective_range:
        return strength * (1/distance_norm - 1/effective_range)
    else:
        return np.array([0])

def repulsive_force_pedestrian(current_pos, obstacle_pos, strength=200.0, effective_range=100):
    # Repulsive force away from the obstacle
    distance = np.array(current_pos) - np.array(obstacle_pos)
    distance_norm = np.linalg.norm(distance)
    print("distance_norm:", distance_norm)
    if distance_norm < effective_range:
        return strength * (1/distance_norm - 1/effective_range)
    else:
        return np.array([0])

class PedestrianPrediction:
    def __init__(self, kf2, kf_1_wrt_2, kf_ped_wrt_car_2, car1_start, car2_start, pedestrian_start, speed_car_1, speed_car_2, pedestrian_speed):
        self.kf_1_wrt_2 = kf_1_wrt_2
        self.KF_2 = kf2
        self.kf_ped_wrt_car_2 = kf_ped_wrt_car_2
        self.car1_x, self.car1_y = car1_start
        self.car2_x, self.car2_y = car2_start
        self.pedestrian_x, self.pedestrian_y = pedestrian_start
        self.speed_car_1 = speed_car_1
        self.speed_car_2 = speed_car_2
        self.pedestrian_speed_x, self.pedestrian_speed_y = pedestrian_speed
        # self.predicted_pos_car_1 = 0
        self.predicted_pos_car_2 = 0
        self.updated_pose_car_1 = 0
        self.transformed_predictions = []
        self.pedestrian_x_loop = 0
        self.pedestrian_y_loop = 0
        self.car_1_x_loop = 0
        self.car_1_y_loop = 0
        self.car_2_x_loop = 0
        self.car_2_y_loop = 0

    def distance_to_pedestrian(self, car_x, car_y):
        return abs(car_y - self.pedestrian_y)

    def get_transformation_matrix(self, dx, dy, theta):
        return np.array([[np.cos(theta), -np.sin(theta), dx],
                         [np.sin(theta), np.cos(theta), dy],
                         [0, 0, 1]])

    def update_positions(self, car1_pos, car2_pos, pedestrian_pose):
        self.car1_x = car1_pos[0] + random.gauss(0, 0.2)
        self.car1_y = car1_pos[1] + random.gauss(0, 0.2)

        self.car2_x = car2_pos[0] + random.gauss(0, 0.2)
        self.car2_y = car2_pos[1] + random.gauss(0, 0.2)

        self.pedestrian_x = pedestrian_pose[0] + random.gauss(0, 0.5)
        self.pedestrian_y = pedestrian_pose[1] + random.gauss(0, 0.5)

    def predict_update_loop(self, dt):

        time_to_collision_car_1 = self.distance_to_pedestrian(self.car1_x, self.car1_y) / abs(self.speed_car_1)
        num_prediction_steps_car_1 = int(time_to_collision_car_1 / 3)

        pos_car_1_wrt_car_2 = np.array(
            [[self.car1_x - self.car2_x], [-self.car1_y + self.car2_y]]) + random.gauss(0, 1)
        pos_car_2 = np.array(
            [[self.car2_x], [self.car2_y]]) + random.gauss(0, 1)
        pose_ped_wrt_car_2 = np.array(
            [[self.pedestrian_x - self.car2_x], [-self.pedestrian_y + self.car2_y]]) + random.gauss(0, 1)

        self.pedestrian_x_loop = self.pedestrian_x
        self.pedestrian_y_loop = self.pedestrian_y
        self.car_1_x_loop = self.car1_x
        self.car_1_y_loop = self.car1_y
        self.car_2_x_loop = self.car2_x
        self.car_2_y_loop = self.car2_y
        for _ in range(num_prediction_steps_car_1):

            self.predicted_pos_car_1_wrt_car_2 = self.kf_1_wrt_2.predict(dt)
            self.predicted_pos_car_2 = self.KF_2.predict(dt)
            self.predicted_ped_wrt_car_2 = self.kf_ped_wrt_car_2.predict(dt)
            self.updated_pose_car_1_wrt_car_2 = self.kf_1_wrt_2.update(pos_car_1_wrt_car_2)
            self.updated_pos_car_2 = self.KF_2.update(pos_car_2)
            self.updated_pose_ped_wrt_car_2 = self.kf_ped_wrt_car_2.update(pose_ped_wrt_car_2)

            pos_car_1_wrt_car_2 = np.array([[self.car_1_x_loop - self.car_2_x_loop], [-self.car_1_y_loop + self.car_2_y_loop]])
            pos_car_2 = np.array([[self.car_2_x_loop], [self.car_2_y_loop]])
            pose_ped_wrt_car_2 = np.array([[self.pedestrian_x_loop - self.car_2_x_loop], [-self.pedestrian_y_loop + self.car_2_y_loop]])

            self.pedestrian_x_loop += self.pedestrian_speed_x * 3
            self.pedestrian_y_loop += self.pedestrian_speed_y * 3

            self.car_1_x_loop += 0 * 3
            self.car_1_y_loop += self.speed_car_1 * 3

            self.car_2_x_loop += 0 * 3
            self.car_2_y_loop += self.speed_car_2 * 3

    def transform(self):
        dx = -self.car_1_x_loop + self.car_2_x_loop
        dy = self.car_1_y_loop - self.car_2_y_loop
        T = self.get_transformation_matrix(dx, dy, 0)

        predicted_ped_pos_in_car_2_homogeneous = np.append(self.predicted_ped_wrt_car_2, [[1]], axis=0)
        transformed_prediction = np.dot(T, predicted_ped_pos_in_car_2_homogeneous)
        transformed_prediction_2d = np.squeeze(transformed_prediction[0:2])

        self.transformed_predictions.append(transformed_prediction_2d)

        return transformed_prediction_2d


class relobjectposition:
    def __init__(self,):
        self.transformed_pos = []
        self.transformed_distance = []
        self.transformed_distance_in_car1 = 0

    def distance_to_obj(self, car_pos, object_pos):
        car_x, car_y = car_pos
        object_x, object_y = object_pos
        return sqrt((car_y - object_y)**2 + (car_x - object_x)**2)

    def get_transformation_matrix(self, dx, dy, theta):
        return np.array([[np.cos(theta), -np.sin(theta), dx],
                         [np.sin(theta), np.cos(theta), dy],
                         [0, 0, 1]])

    def transform(self,  car1_pos, car2_pos, object_pos):

        car1_x = car1_pos[0] + random.gauss(0, 1)
        car1_y = car1_pos[1] + random.gauss(0, 1)
        car2_x = car2_pos[0] + random.gauss(0, 1)
        car2_y = car2_pos[1] + random.gauss(0, 1)
        object_x = object_pos[0] + random.gauss(0, 1)
        object_y = object_pos[1] + random.gauss(0, 1)

        dx = -car1_x + car2_x
        dy = car1_y - car2_y

        T = self.get_transformation_matrix(dx, dy, 0)
        obj_pos_in_car2_homogeneous = np.array([[object_x - car2_x], [car2_y - object_y], [1]])

        transformed_pos = np.dot(T, obj_pos_in_car2_homogeneous)
        transformed_prediction_2d = np.squeeze(transformed_pos[0:2])

        self.transformed_distance_in_car1 = np.linalg.norm(transformed_prediction_2d)

        self.transformed_pos.append(transformed_prediction_2d)
        self.transformed_distance.append(self.transformed_distance_in_car1)

        return self.transformed_distance_in_car1


pedestrian_prediction = PedestrianPrediction(kf2=kf2, kf_1_wrt_2=kf_1_wrt_2, kf_ped_wrt_car_2=kf_ped_wrt_car_2, speed_car_1=speed_car_1, car1_start=(car1_x, car1_y),
                                             car2_start=(car2_x, car2_y), pedestrian_start=(pedestrian_x, pedestrian_y),
                                             speed_car_2=speed_car_2, pedestrian_speed=(pedestrian_speed_x, pedestrian_speed_y))

rel_object_pos = relobjectposition()

while car1_y > pedestrian_prediction.pedestrian_y:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    left_fourth_width = width // 6
    pygame.draw.rect(screen, GRAY, pygame.Rect(0, 0, left_fourth_width, height))
    pygame.draw.rect(screen, BLUE, pygame.Rect(left_fourth_width, 0, width - left_fourth_width, height))

    for y in range(line_y, height, line_height + line_gap):
        pygame.draw.rect(screen, WHITE, (line_x1, y, line_width, line_height))
        pygame.draw.rect(screen, WHITE, (line_x2, y, line_width, line_height))
        pygame.draw.rect(screen, WHITE, (line_x3, y, line_width, line_height))
        pygame.draw.rect(screen, WHITE, (line_x4, y, line_width, line_height))

    screen.blit(car1_image, (car1_x, car1_y))
    screen.blit(car2_image, (car2_x, car2_y))
    screen.blit(rel_object_image, (rel_object_x, rel_object_y))
    screen.blit(pedestrian_image, (pedestrian_x, pedestrian_y))

    # Draw Lines between Cars and Object
    pygame.draw.line(screen, GREEN, (car1_x + car_width/2, car1_y + car_height/2),
                     (rel_object_x + rel_object_width/2, rel_object_y + rel_object_height/2), width=3)

    pygame.draw.line(screen, GREEN, (car2_x + car_width / 2, car2_y + car_height / 2),
                     (rel_object_x + rel_object_width / 2, rel_object_y + rel_object_height / 2), width=3)

    pygame.draw.line(screen, GREEN, (car2_x + car_width / 2, car2_y + car_height / 2),
                     (car1_x + car_width / 2, car1_y + car_height / 2), width=3)

    pygame.draw.line(screen, RED, (car1_x + car_width / 2, car1_y + car_height / 2),
                     (pedestrian_x + pedestrian_width / 2, pedestrian_y + pedestrian_height / 2), width=3)

    pygame.draw.line(screen, RED, (car2_x + car_width / 2, car2_y + car_height / 2),
                     (pedestrian_x + pedestrian_width / 2, pedestrian_y + pedestrian_height / 2), width=3)

    # Draw the trace as black dots
    for pos in car1_trace:
        pygame.draw.circle(screen, BLACK, (int(pos[0]) + car_width//2, int(pos[1]) + car_height//2), 2)  # 3 is the radius of the dots

    # Update positions
    new_car1_pos = (car1_x, car1_y)
    new_car2_pos = (car2_x, car2_y)
    new_pedestrian_pos = (pedestrian_x, pedestrian_y)
    new_object_pos = (rel_object_x, rel_object_y)

    pedestrian_prediction.update_positions(car1_pos=new_car1_pos,
                                           car2_pos=new_car2_pos,
                                           pedestrian_pose=new_pedestrian_pos)

    real_distance_car1 = rel_object_pos.distance_to_obj(car_pos=(car1_x, car1_y),
                                                        object_pos=(rel_object_x, rel_object_y))

    transformed_distance_car1 = rel_object_pos.transform(car1_pos=(car1_x, car1_y), car2_pos=(car2_x, car2_y),
                                                         object_pos=(rel_object_x, rel_object_y))

    error_distance_car1 = abs((real_distance_car1-transformed_distance_car1) / real_distance_car1)
    uncertainty_value = 1 - error_distance_car1
    uncertainty_values.append(uncertainty_value)

    pedestrian_prediction.predict_update_loop(dt=dt*3)
    transformed_pos_of_pedestrian = pedestrian_prediction.transform()

    pedestrian_predicted_global_coordinate_x = \
        pedestrian_prediction.transform()[0, 0] + pedestrian_prediction.predicted_pos_car_2[0, 0] + \
        pedestrian_prediction.predicted_pos_car_1_wrt_car_2[0, 0] + pedestrian_width // 2

    pedestrian_predicted_global_coordinate_y = -pedestrian_prediction.transform()[0, 1] + \
                                               pedestrian_prediction.predicted_pos_car_2[1, 0] - \
                                               pedestrian_prediction.predicted_pos_car_1_wrt_car_2[1, 0] + \
                                               pedestrian_width // 2

    circle_center = (pedestrian_predicted_global_coordinate_x, pedestrian_predicted_global_coordinate_y)
    circle_radius = sqrt(2) * pedestrian_width // 2
    halo_radius = circle_radius//(uncertainty_value * 2)
    draw_halo(screen, circle_center, halo_radius, RED, 120)

    car1_y += speed_car_1
    car2_y += speed_car_2
    pedestrian_x += pedestrian_speed_x
    pedestrian_y += pedestrian_speed_y
    rel_object_x += rel_object_speed

    real_positions_car_1.append(new_pedestrian_pos[0] - new_car1_pos[0])
    real_positions_car_2.append(new_pedestrian_pos[0] - new_car2_pos[0])
    estimations_car_1.append(pedestrian_prediction.transform()[0, 0])
    estimations_car_2.append(pedestrian_prediction.predicted_pos_car_2[0, 0])

    # Inside the loop, after updating Kalman Filter estimations
    squared_diff_car_1 = (pedestrian_prediction.transform()[0, 0] - (new_pedestrian_pos[0] - new_car1_pos[0])) ** 2
    squared_diff_car_2 = (pedestrian_prediction.predicted_pos_car_2[0, 0] - (new_pedestrian_pos[0] - new_car2_pos[0])) ** 2

    squared_diffs_car_1.append(squared_diff_car_1)
    squared_diffs_car_2.append(squared_diff_car_2)

    goal_pos = [7 * width // 12 - car_width // 2, 0]  # Goal is to reach y = 0
    goal_force = attractive_force((car1_x, car1_y), goal_pos)
    pedestrian_force = repulsive_force_pedestrian((car1_x, car1_y), (pedestrian_predicted_global_coordinate_x,
                                                                     pedestrian_predicted_global_coordinate_y))
    car2_force = repulsive_force_car((car1_x, car1_y), (car2_x, car2_y))

    total_force = goal_force + pedestrian_force + car2_force
    car1_x += total_force[0]

    car1_trace.append((car1_x, car1_y))

    # Update the display
    frame = pygame.surfarray.array3d(screen)
    frame = frame.transpose([1, 0, 2])
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video.write(frame)

    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

transformed_x_positions = [pos[0, 0] for pos in pedestrian_prediction.transformed_predictions]


plt.figure(1)
plt.plot(real_positions_car_1, label='Real Position')
plt.plot(estimations_car_1, label='Estimation Car 1')
plt.plot(transformed_x_positions, label='Transformed Prediction')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Real Position vs Estimation for Car 1')
plt.legend()

plt.figure(2)
plt.plot(real_positions_car_2, label='Real Position')
plt.plot(estimations_car_2, label='Estimation Car 2')
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Real Position vs Estimations for Car 2')
plt.legend()

plt.figure(3)
plt.plot(squared_diffs_car_1, label='Error Value')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Error Value for the Prediction of Car 1')
plt.legend()

plt.figure(4)
plt.plot(squared_diffs_car_2, label='Error Value')
plt.xlabel('Time Step')
plt.ylabel('Error')
plt.title('Error Value for the Prediction of Car 2')
plt.legend()

plt.figure(5)
plt.plot(uncertainty_values, label='Uncertainty Value')
plt.xlabel('Time Step')
plt.ylabel('Uncertainty')
plt.title('Uncertainty Value for the distance of Car 1')
plt.legend()

plt.show()
video.release()
pygame.quit()
sys.exit()