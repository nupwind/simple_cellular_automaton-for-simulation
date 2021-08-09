import numpy as np
import random
import copy
import matplotlib.pyplot as plt


# one entrance
# def initialization(place_length, place_width, exit_point_position, exit_query_length):
    # people_distribution = np.zeros((place_length, place_width), dtype=np.int)
    # for i, w in enumerate(exit_point_position, 0):
    #     for h in range(exit_query_length[i]):
    #         people_distribution[h][w] = 1
    # return people_distribution

 
def static_field(place_length, place_width, exit_point_position): 
    static_distance = np.zeros((place_length, place_width), dtype=np.longdouble)
    for h in range(place_length):
        for w in range(place_width):
            distance_temp = np.longdouble(500000)
            for i, exit_position in enumerate(exit_point_position, 0):
                _distance = np.sqrt(np.longdouble(h)**2 + np.longdouble(exit_position-w)**2)
                if _distance < distance_temp: distance_temp = _distance
            static_distance[h][w] = distance_temp
    return static_distance


def dynamic_field(people_distribution, k_d):
    H = len(people_distribution)
    W = len(people_distribution[0])
    dynamic_distance = np.zeros((H, W), dtype=np.float32)
    for h in range(len(people_distribution)):
        for w in range(len(people_distribution[h])):
            # print(h, w)
            if h == 0: 
                dynamic_distance[h][w] = 500000     # a large number
            elif h == H-1:
                if w == 0: 
                    dynamic_distance[h][w] = people_distribution[H-2][0] + people_distribution[H-1][1] + people_distribution[H-2][1]
                elif w == W-1: 
                    dynamic_distance[h][w] = people_distribution[H-2][W-1] + people_distribution[H-2][W-2] + people_distribution[H-1][W-2]
                else: 
                    dynamic_distance[h][w] = people_distribution[h-1][w] + people_distribution[h-1][w-1] + people_distribution[h-1][w] \
                                                + people_distribution[h-1][w+1] + people_distribution[h][w+1]
            else:
                if w == 0:
                    dynamic_distance[h][w] = people_distribution[h-1][w] + people_distribution[h+1][w] + people_distribution[h-1][w+1] \
                                                + people_distribution[h][w+1] + people_distribution[h+1][w+1]
                elif w == W-1:
                    dynamic_distance[h][w] = people_distribution[h-1][w] + people_distribution[h+1][w] + people_distribution[h-1][w-1] \
                                                + people_distribution[h-1][w-1] + people_distribution[h-1][w-1]
                else:
                    dynamic_distance[h][w] = people_distribution[h-1][w-1] + people_distribution[h-1][w] + people_distribution[h-1][w+1] \
                                                + people_distribution[h][w-1] + people_distribution[h][w+1] \
                                                + people_distribution[h+1][w-1] + people_distribution[h+1][w] + people_distribution[h+1][w+1]
    # P = k_d * dynamic_distance
    # print(P[:20, 19:20])
    # assert 0
    return dynamic_distance


def _update_output(people_distribution_up, remaining_num, timer, FORBIDDEN_AREA_LENGTH, EXIT_POINT_POSITION):
    if timer % 10 == 0 and timer != 0:       # update exit
        for w in range(len(remaining_num)):
            remaining_num[w] += 1 
            if remaining_num[w] >= 20:
                remaining_num[w] = 20
            people_distribution_up[FORBIDDEN_AREA_LENGTH[0]-remaining_num[w], EXIT_POINT_POSITION[w]] = 0  
    return people_distribution_up, remaining_num


def _update_input(people_distribution, ENTRANCE_POSITION, timer): 
    if timer % 10 == 0:  
        people_distribution[-1, ENTRANCE_POSITION:] = 1
    return people_distribution


def _move(people_distribution, whole_distance_distribution, people_distribution_up, EXIT_POINT_POSITION, remaining_num, FORBIDDEN_AREA_LENGTH):  # forbidden_area_length = 20
    H = len(people_distribution)
    W = len(people_distribution[0])

    for h in range(H):
        for w in range(W):
            if people_distribution[h, w] == 0: pass
            else:
                # 尝试进入排队
                # 边界情况 往中心走，优先选择斜着走 再选择水平 再选择竖直，三种行为方式选择一个。
                # four boundary
                if h == 1: ## 走不到h=0的区域 不知道为什么 is a bug
                    if w in EXIT_POINT_POSITION:    
                        idx = EXIT_POINT_POSITION.index(w)
                        if remaining_num[idx] > 0:
                            remaining_num[idx] -= 1
                            people_distribution[h, w] = 0  
                            people_distribution_up[FORBIDDEN_AREA_LENGTH[0] - remaining_num[idx] - 1, w] = 1   
                        else: 
                            pass
                    else:
                        if w <= (W-1)/2:
                            if people_distribution[h+1, w+1] == 0:
                                people_distribution[h+1, w+1] = 1
                                people_distribution[h, w] = 0
                            elif people_distribution[h, w+1] == 0:
                                people_distribution[h, w+1] = 1
                                people_distribution[h, w] = 0
                            elif people_distribution[h+1, w] == 0:
                                people_distribution[h+1, w] = 1
                                people_distribution[h, w] = 0
                            else:
                                pass
                        else: 
                            if people_distribution[h-1, w-1] == 0:
                                people_distribution[h-1, w-1] = 1
                                people_distribution[h, w] = 0
                            elif people_distribution[h, w-1] == 0:
                                people_distribution[h, w-1] = 1
                                people_distribution[h, w] = 0
                            elif people_distribution[h-1, w] == 0:
                                people_distribution[h-1, w] = 1
                                people_distribution[h, w] = 0
                            else:
                                pass                 
                elif h == H-1:
                    if w <= (W-1)/2:
                        if people_distribution[h-1, w+1] == 0:
                            people_distribution[h-1, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w+1] == 0:
                            people_distribution[h, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h-1, w] == 0:
                            people_distribution[h-1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                    else: 
                        if people_distribution[h-1, w-1] == 0:
                            people_distribution[h-1, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w-1] == 0:
                            people_distribution[h, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h-1, w] == 0:
                            people_distribution[h-1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                elif w == 0:
                    if h <= (H-1)/2:
                        if people_distribution[h+1, w+1] == 0:
                            people_distribution[h+1, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w+1] == 0:
                            people_distribution[h, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h+1, w] == 0:
                            people_distribution[h+1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                    else: 
                        if people_distribution[h-1, w+1] == 0:
                            people_distribution[h-1, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w+1] == 0:
                            people_distribution[h, w+1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h-1, w] == 0:
                            people_distribution[h-1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                elif w == W-1:
                    if h <= (H-1)/2:
                        if people_distribution[h+1, w-1] == 0:
                            people_distribution[h+1, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w-1] == 0:
                            people_distribution[h, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h-1, w] == 0:
                            people_distribution[h-1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                    else: 
                        if people_distribution[h-1, w-1] == 0:
                            people_distribution[h-1, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h, w-1] == 0:
                            people_distribution[h, w-1] = 1
                            people_distribution[h, w] = 0
                        elif people_distribution[h-1, w] == 0:
                            people_distribution[h-1, w] = 1
                            people_distribution[h, w] = 0
                        else:
                            pass
                else:
                    # not boundary 8 value comparation
                    trans_probability_group = np.array(
                        (whole_distance_distribution[h-1, w-1], whole_distance_distribution[h-1, w],whole_distance_distribution[h-1, w+1], 
                           whole_distance_distribution[h, w-1], whole_distance_distribution[h, w+1],  
                           whole_distance_distribution[h+1, w-1], whole_distance_distribution[h+1, w],whole_distance_distribution[h+1, w+1]), dtype=np.longdouble) 
                    if_trans = False
                    while not if_trans:
                        idx = np.argmax(trans_probability_group)
                        if np.max(trans_probability_group) == 0:
                            break
                        else:
                            # can transform successfully one time 
                            if idx == 0 and people_distribution[h-1, w-1] == 0:
                                people_distribution[h-1, w-1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True
                            elif idx == 1 and people_distribution[h-1, w] == 0:
                                people_distribution[h-1, w] = 1
                                people_distribution[h, w] = 0   
                                if_trans = True              
                            elif idx == 2 and people_distribution[h-1, w+1] == 0:
                                people_distribution[h-1, w+1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True   
                            elif idx == 3 and people_distribution[h, w-1] == 0:
                                people_distribution[h, w-1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True     
                            elif idx == 4 and people_distribution[h, w+1] == 0:
                                people_distribution[h, w+1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True         
                            elif idx == 5 and people_distribution[h+1, w-1] == 0:
                                people_distribution[h+1, w-1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True     
                            elif idx == 6 and people_distribution[h+1, w] == 0:
                                people_distribution[h+1, w] = 1
                                people_distribution[h, w] = 0
                                if_trans = True           
                            elif idx == 7 and people_distribution[h+1, w+1] == 0:
                                people_distribution[h+1, w+1] = 1
                                people_distribution[h, w] = 0
                                if_trans = True      
                            else:
                                # second transfer
                                trans_probability_group[idx] = 0
    return people_distribution, people_distribution_up, remaining_num


def main():
    EPOCH = 1000
    # whole place size is (FORBIDDEN_AREA_LENGTH + PLACE_LENGTH) * PLACE_WIDTH
    FORBIDDEN_AREA_LENGTH = [20, 20, 20, 20, 20]
    PLACE_LENGTH = 30
    PLACE_WIDTH = 100
    EXIT_POINT_NUM = 5
    EXIT_POINT_POSITION = [10, 30, 50, 70, 90]
    EXIT_QUERY_LEHGTH = [5, 9, 13, 17, 20]
    ENTRANCE_POSITION = 90    # 90 TO 100
    ## k 越小效果越强
    k_d = -3  # dynamic field parameter
    k_s = -0.2   # static field patameter
    timer = 0

    people_distribution = np.zeros((PLACE_LENGTH, PLACE_WIDTH), dtype=np.int)
    static_field_distribution = static_field(PLACE_LENGTH, PLACE_WIDTH, EXIT_POINT_POSITION)
    # print(static_field_distribution)
    # plt.imshow(static_field_distribution)
    # plt.show()
    # assert 0
    people_distribution_up = np.zeros((FORBIDDEN_AREA_LENGTH[0], PLACE_WIDTH), dtype=np.int)
    remaining_num  = []
    for i in range(EXIT_POINT_NUM):
        remaining_num.append(np.max(FORBIDDEN_AREA_LENGTH[i] - EXIT_QUERY_LEHGTH[i], 0))
    
    # make people_distribution_up
    exit_queue_length = []
    for i, w_position in enumerate(EXIT_POINT_POSITION):
        length_temp = 0 
        exit_queue_length.append(np.max(FORBIDDEN_AREA_LENGTH[0] - remaining_num[i], 0))
        while length_temp < exit_queue_length[i]:
            # print(length_temp, w_position)
            people_distribution_up[length_temp, w_position] = 1
            length_temp += 1
    # plt.imshow(people_distribution_up)
    # plt.pause(0.5) 
    # plt.show()

    ## get start
    ###############################
    for i in range(EPOCH):
        print("timer: ", timer, 'remaining_num', remaining_num)

        whole_people_distribution = np.concatenate((people_distribution_up, people_distribution), axis=0)
        plt.imshow(whole_people_distribution)
        plt.pause(0.1) 
        # plt.show()

        '''
            1、update output people and input people
            2、calculate dynamic_field
            3、calculate distance(2 fields)
            4、move: update people_distribution:
        '''
        # update output people
        people_distribution_up, remaining_num = _update_output(people_distribution_up, remaining_num, timer, FORBIDDEN_AREA_LENGTH, EXIT_POINT_POSITION)
        # update input people
        people_distribution = _update_input(people_distribution, ENTRANCE_POSITION, timer)
        # calculate dynamic_field
        D = dynamic_field(people_distribution, k_d)
        S = static_field_distribution
        # print(D.shape)
        # plt.imshow(D)
        # plt.show()
        # assert 0
        # calculate distance(2 fields)
        whole_distance_distribution = np.exp(k_d * D + k_s * S)
        # move: update people_distribution
        people_distribution, people_distribution_up, remaining_num = _move(people_distribution, whole_distance_distribution, people_distribution_up,  
                                                                                EXIT_POINT_POSITION, remaining_num, FORBIDDEN_AREA_LENGTH)

        timer += 1

 
if __name__ == '__main__':
    main()


