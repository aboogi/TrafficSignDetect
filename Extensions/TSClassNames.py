def traffic_signs_names(class_num):
        class_ts = {
            0: 'Speed Limit 20 km/h',
            1: 'Speed Limit 30 km/h',
            2: 'Speed Limit 50 km/h',
            3: 'Speed Limit 60 km/h',
            4: 'Speed Limit 70 km/h',
            5: 'Speed Limit 80 km/h',
            6: 'End of Speed Limit 80 km/h',
            7: 'Speed Limit 100 km/h',
            8: 'Speed Limit 110 km/h',
            9: 'No overtaking',
            10: 'No overtaking by trucks',
            11: 'Right-of-way at the next intersection',
            12: 'Priority road',
            13: 'Yield sign',
            14: 'Stop',
            15: 'No vechiles',
            16: 'No entry for trucks',
            17: 'No entry',
            18: 'General caution',
            19: 'Dangerous curve to the left',
            20: 'Dangerous curve to the right',
            21: 'Double curve',
            22: 'Bumpy road',
            23: 'Slippery road',
            24: 'Road narrows on the right',
            25: 'Roadworks',
            26: 'Traffic signals',
            27: 'Pedestrians',
            28: 'Children crossing',
            29: 'Bicycles crossing',
            30: 'Beware of ice/snow',
            31: 'Wild animals crossing',
            32: 'End of all speed and passing limits',
            33: 'Turn right ahead',
            34: 'Turn left ahead',
            35: 'Ahead only',
            36: 'Go straight or right',
            37: 'Go straight or left',
            38: 'Keep right',
            39: 'Keep left',
            40: 'Roundabout mandatory',
            41: 'End of no passing',
            42: 'End of no passing by vechiles over 3.5 metric tons',
        }
        if class_num[0] in class_ts:
            return class_ts[class_num[0]]


if __name__ == '__main__':
    l = traffic_signs_names('40')
    print(l)
