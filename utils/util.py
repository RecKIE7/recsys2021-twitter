import math


def get_day_phase(hour):
    night = [22,23,24,0,1,2,3,4,5]
    morining = [6,7,8,9,10,11]
    lunch = [12,13,14]
    afternoon = [15,16,17,18]
    evening = [19,20,21]
    if hour in night:
        return "night"
    elif hour in morining:
        return "morning"
    elif hour in lunch:
        return "lunch"
    elif hour in afternoon:
        return "afternoon"
    elif hour in evening:
        return "evening"

def get_hours_from_midnight_of_current_day(timestamp):
    fixed_point = timestamp.replace(hour=0,minute=0,second=0)
    difference = timestamp - fixed_point
    return math.floor(difference.seconds/60/60)