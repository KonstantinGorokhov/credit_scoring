import random


def generate_hidden_features():
    appl_rej_cnt = random.randint(0, 5)

    if appl_rej_cnt <= 1:
        score_bki = round(random.uniform(-3.0, -1.0), 2)
    else:
        score_bki = round(random.uniform(-1.0, 2.0), 2)

    return {
        "appl_rej_cnt": appl_rej_cnt,
        "Score_bki": score_bki,
        "out_request_cnt": random.randint(0, 5),
        "region_rating": random.choice([20, 30, 40, 50, 60, 70, 80]),
        "home_address_cd": random.choice([1, 2]),
        "work_address_cd": random.choice([1, 2, 3]),
        "SNA": random.choice([1, 2, 3, 4]),
        "first_time_cd": random.randint(1, 5)
    }
