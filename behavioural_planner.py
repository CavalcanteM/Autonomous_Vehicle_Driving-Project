#!/usr/bin/env python3
import numpy as np
import math
import logging

# State machine states
FOLLOW_LANE = 0
DECELERATE_TO_STOP = 1
STAY_STOPPED = 2
# Stop speed threshold
STOP_THRESHOLD = 0.02
# EditGroup2
DISTANCE_FROM_TRAFFIC_LIGHT = 2
# EndEditGroup2

class BehaviouralPlanner:
    def __init__(self, lookahead, lead_vehicle_lookahead):
        self._lookahead                     = lookahead
        self._follow_lead_vehicle_lookahead = lead_vehicle_lookahead
        self._state                         = FOLLOW_LANE
        self._follow_lead_vehicle           = False
        self._obstacle_on_lane              = False
        self._goal_state                    = [0.0, 0.0, 0.0]
        self._goal_pedestrian               = []    # EditGroup2
        self._goal_index                    = 0
        self._traffic_light_fences          = []    # EditGroup2
        self._is_traffic_light_green        = True  # EditGroup2
    
    def set_lookahead(self, lookahead):
        self._lookahead = lookahead

    def set_is_traffic_light_green(self, value):
        self._is_traffic_light_green = value

    def add_traffic_light_fences(self, traffic_light_fences):
        for fence in traffic_light_fences:
            self._traffic_light_fences.insert(0, fence)

    def get_follow_lead_vehicle(self):
        return self._follow_lead_vehicle

    # Handles state transitions and computes the goal state.
    def transition_state(self, waypoints, ego_state, closed_loop_speed):
        """Handles state transitions and computes the goal state.  
        
        args:
            waypoints: current waypoints to track (global frame). 
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closed_loop_speed: current (closed-loop) speed for vehicle (m/s)
        variables to set:
            self._goal_index: Goal index for the vehicle to reach
                i.e. waypoints[self._goal_index] gives the goal waypoint
            self._goal_state: Goal state for the vehicle to reach (global frame)
                format: [x_goal, y_goal, v_goal]
            self._state: The current state of the vehicle.
                available states: 
                    FOLLOW_LANE         : Follow the global waypoints (lane).
                    DECELERATE_TO_STOP  : Decelerate to stop.
                    STAY_STOPPED        : Stay stopped.
            self._traffic_light_fences: Contains the list of all the fences found for a traffic light
                format: [[x0_i, y0_i, x0_f, y0_f],
                         [x1_i, y1_i, x1_f, y1_f],
                         ...
                         [xn_i, yn_i, xn_f, yn_f]]
        useful_constants:
            STOP_THRESHOLD  : Stop speed threshold (m). The vehicle should fully
                              stop when its speed falls within this threshold.
        """
        # In this state, continue tracking the lane by finding the
        # goal index in the waypoint list that is within the lookahead
        # distance. Then, check to see if the waypoint path intersects
        # with any traffic light lines. If it does, then ensure that the goal
        # state enforces the car to be stopped before the traffic light line.
        # You should use the get_closest_index(), get_goal_index(), and
        # check_for_traffic_lights() helper functions.
        # Make sure that get_closest_index() and get_goal_index() functions are
        # complete, and examine the check_for_traffic_lights() function to
        # understand it.
        if self._state == FOLLOW_LANE:
            # First, find the closest index to the ego vehicle.
            closest_len, closest_index = get_closest_index(waypoints, ego_state)

            # Next, find the goal index that lies within the lookahead distance
            # along the waypoints.
            goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
            while waypoints[goal_index][2] <= 0.1: goal_index += 1

            # Successivamente controlliamo se ci sono ostacoli o semafori rossi, in caso affermativo modifichiamo il _goal_state
            # per permettere al veicolo di fermarsi nel punto desiderato
            if len(self._traffic_light_fences) > 0:
                goal_traffic_light, traffic_light_found = self.check_for_traffic_lights(waypoints, closest_index, goal_index, ego_state)
            else:
                traffic_light_found = False
            
            # C'?? un semaforo rosso e non ci sono pedoni sulla lane
            # passiamo nello stato DECELERATE_TO_STOP e modifichiamo il goal state
            # per fermarci prima del semaforo
            if traffic_light_found and not self._obstacle_on_lane:
                self._goal_state = goal_traffic_light
                self._state = DECELERATE_TO_STOP
                self._traffic_light_fences.clear()
                logging.debug("TRAFFIC LIGHT FOUND")
                logging.info("FOLLOW_LANE => DECELERATE_TO_STOP")

            # Non c'?? un semaforo rosso, ma c'?? un pedone sulla lane, passiamo nello
            # stato DECELERATE_TO_STOP e modifichiamo il goal state per fermarci
            # prima del pedone e non investirlo
            elif self._obstacle_on_lane and not traffic_light_found:
                self._goal_state = self._goal_pedestrian
                self._state = DECELERATE_TO_STOP
                logging.debug("PEDESTRIAN FOUND")
                logging.info("FOLLOW_LANE => DECELERATE_TO_STOP")

            # C'?? sia un semaforo rosso sia un pedone sulla lane, controlliamo quale dei due
            # ?? pi?? vicino al nostro veicolo e modifichiamo il goal state per fermarci prima
            # del pedone/semaforo
            elif self._obstacle_on_lane and traffic_light_found:
                self._goal_state = self.min_distance(ego_state, goal_traffic_light, self._goal_pedestrian)
                self._state = DECELERATE_TO_STOP
                if self._goal_state == goal_traffic_light:
                    self._traffic_light_fences.clear()
                    logging.debug("BOTH FOUND, BUT TRAFFIC LIGHT")
                else:
                    logging.debug("BOTH FOUND, BUT PEDESTRIAN")
                logging.info("FOLLOW_LANE => DECELERATE_TO_STOP")

            # Non c'?? un semaforo rosso o un pedone sulla lane, quindi possiamo
            # continuare il nostro percorso seguendo i waypoint
            else:
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]


        # In this state, check if we have reached a complete stop. Use the
        # closed loop speed to do so, to ensure we are actually at a complete
        # stop, and compare to STOP_THRESHOLD.  If so, transition to the next
        # state.
        elif self._state == DECELERATE_TO_STOP:
            
            # Le situazioni possibili sono 3:
            # - Stiamo decelerando per fermarci ad un semaforo e un pedone attraversa tra noi e il semaforo;
            #   in questo caso ci dobbiamo fermare prima del pedone per non investirlo
            # - Un pedone sta attraversando la strada e un altro pedone inizia ad attraversare prima del precedente
            #   in questo caso ci dobbiamo fermare prima del pedone pi?? vicino
            # - Se un pedone cambia direzione ?? possibile che il punto di collisione cambi, quindi dobbiamo controllare
            #   se questo adesso non sia prima di quello calcolato precedentemente
            if self._obstacle_on_lane:
                self._goal_state = self.min_distance(ego_state, self._goal_state, self._goal_pedestrian)

            # Se non abbiamo un pedone sulla lane e non abbiamo semafori rossi in vista
            # possiamo passare nello stato di FOLLOW_LANE
            elif not self._obstacle_on_lane and self._is_traffic_light_green:
                self._state = FOLLOW_LANE
                logging.info("DECELERATE_TO_STOP => FOLLOW_LANE")

            # Se la nostra velocit?? attuale ?? scesa sotto la STOP_THRESHOLD
            # possiamo ritenere il veicolo fermo e passare nello stato di STAY_STOPPED
            elif abs(closed_loop_speed) <= STOP_THRESHOLD:
                self._state = STAY_STOPPED
                logging.info("DECELERATE_TO_STOP => STAY_STOPPED")

        # In this state, check to see if we have stayed stopped for at
        # least STOP_COUNTS number of cycles. If so, we can now leave
        # the stop sign and transition to the next state.
        elif self._state == STAY_STOPPED:
            # Allow the ego vehicle to leave the traffic light. Once it has
            # passed the traffic light, return to lane following.
            # You should use the get_closest_index(), get_goal_index(), and 
            # helper functions.

            # Se non ci sono pedoni sulla lane e non c'?? un semaforo rosso, possiamo
            # passare nello stato di FOLLOW_LANE
            if not self._obstacle_on_lane and self._is_traffic_light_green: 
                closest_len, closest_index = get_closest_index(waypoints, ego_state)
                goal_index = self.get_goal_index(waypoints, ego_state, closest_len, closest_index)
                while waypoints[goal_index][2] <= 0.1: goal_index += 1               
                self._goal_index = goal_index
                self._goal_state = waypoints[goal_index]

                self._state = FOLLOW_LANE
                logging.info("STAY_STOPPED => FOLLOW_LANE")
                
        else:
            raise ValueError('Invalid state value.')

    # EditGroup2
    # Checks the given segment of the waypoint list to see if it
    # intersects with a traffic light line. If any index does, return the
    # new goal state accordingly.
    def check_for_traffic_lights(self, waypoints, closest_index, goal_index, ego_state):
        """Checks for a traffic light that is intervening the goal path.

        Checks for a traffic light that is intervening the goal path. Returns a new
        goal index (the current goal index is obstructed by a traffic light line), and a
        boolean flag indicating if a traffic light obstruction was found.
        
        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
            goal_index (current): Current goal index for the vehicle to reach
                i.e. waypoints[goal_index] gives the goal waypoint
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s) 
        variables to set:
            [goal_index (updated), stop_sign_found]: 
                goal_index (updated): Updated goal index for the vehicle to reach
                    i.e. waypoints[goal_index] gives the goal waypoint
                traffic_light_found: Boolean flag for whether a traffic_light was found or not
        """
        for i in range(closest_index, goal_index):
            # Check to see if path segment crosses any of the traffic light lines.
            intersect_flag = False
            wp_1 = [ego_state[0], ego_state[1]]
            for traffic_light_fence in self._traffic_light_fences:
                wp_2   = np.array(waypoints[i+1][0:2])
                s_1    = np.array(traffic_light_fence[0:2])
                s_2    = np.array(traffic_light_fence[2:4])
                
                v1     = np.subtract(wp_2, wp_1)
                v2     = np.subtract(s_1, wp_2)
                sign_1 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(s_2, wp_2)
                sign_2 = np.sign(np.cross(v1, v2))

                v1     = np.subtract(s_2, s_1)
                v2     = np.subtract(wp_1, s_2)
                sign_3 = np.sign(np.cross(v1, v2))
                v2     = np.subtract(wp_2, s_2)
                sign_4 = np.sign(np.cross(v1, v2))

                # Check if the line segments intersect.
                if (sign_1 != sign_2) and (sign_3 != sign_4):
                    intersect_flag = True

                # Check if the collinearity cases hold.
                if (sign_1 == 0) and pointOnSegment(wp_1, s_1, wp_2):
                    intersect_flag = True
                if (sign_2 == 0) and pointOnSegment(wp_1, s_2, wp_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_1, s_2):
                    intersect_flag = True
                if (sign_3 == 0) and pointOnSegment(s_1, wp_2, s_2):
                    intersect_flag = True

                # If there is an intersection with a traffic light line, update
                # the goal state to stop before the goal line.
                
                if intersect_flag:
                    logging.info("Intersect %s", str(intersect_flag))
                    goal_index = i
                    current_yaw = ego_state[2]
                    if np.sign(round(np.cos(current_yaw))) > 0: # mi sto muovendo lungo le x positive
                        return [traffic_light_fence[0] - DISTANCE_FROM_TRAFFIC_LIGHT, ego_state[1], 0], True
                    elif np.sign(round(np.cos(current_yaw))) < 0: # mi sto muovendo lungo le x negative
                        return [traffic_light_fence[0] + DISTANCE_FROM_TRAFFIC_LIGHT, ego_state[1], 0], True
                    else:
                        if np.sign(round(np.sin(current_yaw))) > 0: # mi sto muovendo lungo le y positive
                            return [ego_state[0], traffic_light_fence[1] - DISTANCE_FROM_TRAFFIC_LIGHT, 0], True
                        else:   # mi sto muovendo lungo le y negative
                            return [ego_state[0], traffic_light_fence[1] + DISTANCE_FROM_TRAFFIC_LIGHT, 0], True

        return goal_index, False
    # EndEditGroup2

    # Gets the goal index in the list of waypoints, based on the lookahead and
    # the current ego state. In particular, find the earliest waypoint that has accumulated
    # arc length (including closest_len) that is greater than or equal to self._lookahead.
    def get_goal_index(self, waypoints, ego_state, closest_len, closest_index):
        """Gets the goal index for the vehicle. 
        
        Set to be the earliest waypoint that has accumulated arc length
        accumulated arc length (including closest_len) that is greater than or
        equal to self._lookahead.

        args:
            waypoints: current waypoints to track. (global frame)
                length and speed in m and m/s.
                (includes speed to track at each x,y location.)
                format: [[x0, y0, v0],
                         [x1, y1, v1],
                         ...
                         [xn, yn, vn]]
                example:
                    waypoints[2][1]: 
                    returns the 3rd waypoint's y position

                    waypoints[5]:
                    returns [x5, y5, v5] (6th waypoint)
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
        returns:
            wp_index: Goal index for the vehicle to reach
                i.e. waypoints[wp_index] gives the goal waypoint
        """
        # Find the farthest point along the path that is within the
        # lookahead distance of the ego vehicle.
        # Take the distance from the ego vehicle to the closest waypoint into
        # consideration.

        arc_length = closest_len
        wp_index = closest_index

        # In this case, reaching the closest waypoint is already far enough for
        # the planner.  No need to check additional waypoints.
        if arc_length > self._lookahead:
            return wp_index

        # We are already at the end of the path.
        if wp_index == len(waypoints) - 1:
            return wp_index

        # Otherwise, find our next waypoint.
        while wp_index < len(waypoints) - 1:
            arc_length += np.sqrt((waypoints[wp_index][0] - waypoints[wp_index+1][0])**2 + (waypoints[wp_index][1] - waypoints[wp_index+1][1])**2)
            if arc_length > self._lookahead: break
            wp_index += 1

        return wp_index % len(waypoints)
    
    # EditGroup2
    # Checks to see if we need to modify our velocity profile to accomodate the
    # lead vehicle.
    def check_for_lead_vehicle(self, ego_state, lead_car_position, lead_car_yaw):
        """Checks for lead vehicle within the proximity of the ego car, such
        that the ego car should begin to follow the lead vehicle.

        args:
            ego_state: ego state vector for the vehicle. (global frame)
                format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                    ego_x and ego_y     : position (m)
                    ego_yaw             : top-down orientation [-pi to pi]
                    ego_open_loop_speed : open loop speed (m/s)
            lead_car_position: The [x, y] position of the lead vehicle.
                Lengths are in meters, and it is in the global frame.
            lead_car_yaw: Yaw of the lead car in degrees
        sets:
            self._follow_lead_vehicle: Boolean flag on whether the ego vehicle
                should follow (true) the lead car or not (false).
        """
        # Check lead car position delta vector relative to heading, as well as
        # distance, to determine if car should be followed.
        # Check to see if lead vehicle is within range, and is ahead of us.
        self._follow_lead_vehicle = False
        min_distance = 1000
        min_index = None
        for i in range(len(lead_car_position)):
            # Compute the angle between the normalized vector between the lead vehicle
            # and ego vehicle position with the ego vehicle's heading vector.
            lead_car_delta_vector = [lead_car_position[i][0] - ego_state[0], 
                                        lead_car_position[i][1] - ego_state[1]]
            lead_car_distance = np.linalg.norm(lead_car_delta_vector)
            # In this case, the car is a possible lead vehicle.   
            if lead_car_distance < self._follow_lead_vehicle_lookahead:
                lead_car_delta_vector = np.divide(lead_car_delta_vector, 
                                                    lead_car_distance)
                ego_heading_vector = [math.cos(ego_state[2]), 
                                        math.sin(ego_state[2])]
                # Check to see if the relative angle between the lead vehicle and the ego
                # vehicle lies within +/- 45 degrees of the ego vehicle's heading.
                if np.dot(lead_car_delta_vector, ego_heading_vector) > (1 / math.sqrt(2)) and \
                    round(math.cos(lead_car_yaw[i])) == round(math.cos(ego_state[2])) and \
                    round(math.sin(lead_car_yaw[i])) == round(math.sin(ego_state[2])):
                    # We want to find the car at minimum distance from us
                    if min_distance > lead_car_distance:
                        min_distance = lead_car_distance
                        min_index = i
                        self._follow_lead_vehicle = True
        return min_index

    # Determina quale di due goal sia il pi?? vicino al nostro veicolo
    # identificato dall'ego_state
    def min_distance(self, ego_state, goal_1, goal_2):
        dist1 = np.linalg.norm([ego_state[0] - goal_1[0], 
                                    ego_state[1] - goal_1[1]])
        dist2 = np.linalg.norm([ego_state[0] - goal_2[0], 
                                    ego_state[1] - goal_2[1]])
        
        if dist1 < dist2:
            return goal_1
        else:
            return goal_2

    # EndEditGroup2


# Compute the waypoint index that is closest to the ego vehicle, and return
# it as well as the distance from the ego vehicle to that waypoint.
def get_closest_index(waypoints, ego_state):
    """Gets closest index a given list of waypoints to the vehicle position.

    args:
        waypoints: current waypoints to track. (global frame)
            length and speed in m and m/s.
            (includes speed to track at each x,y location.)
            format: [[x0, y0, v0],
                     [x1, y1, v1],
                     ...
                     [xn, yn, vn]]
            example:
                waypoints[2][1]: 
                returns the 3rd waypoint's y position

                waypoints[5]:
                returns [x5, y5, v5] (6th waypoint)
        ego_state: ego state vector for the vehicle. (global frame)
            format: [ego_x, ego_y, ego_yaw, ego_open_loop_speed]
                ego_x and ego_y     : position (m)
                ego_yaw             : top-down orientation [-pi to pi]
                ego_open_loop_speed : open loop speed (m/s)

    returns:
        [closest_len, closest_index]:
            closest_len: length (m) to the closest waypoint from the vehicle.
            closest_index: index of the waypoint which is closest to the vehicle.
                i.e. waypoints[closest_index] gives the waypoint closest to the vehicle.
    """
    closest_len = float('Inf')
    closest_index = 0

    for i in range(len(waypoints)):
        temp = (waypoints[i][0] - ego_state[0])**2 + (waypoints[i][1] - ego_state[1])**2
        if temp < closest_len:
            closest_len = temp
            closest_index = i
    closest_len = np.sqrt(closest_len)

    return closest_len, closest_index

# Checks if p2 lies on segment p1-p3, if p1, p2, p3 are collinear.        
def pointOnSegment(p1, p2, p3):
    if (p2[0] <= max(p1[0], p3[0]) and (p2[0] >= min(p1[0], p3[0])) and \
       (p2[1] <= max(p1[1], p3[1])) and (p2[1] >= min(p1[1], p3[1]))):
        return True
    else:
        return False
