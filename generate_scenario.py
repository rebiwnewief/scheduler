from itertools import chain, combinations

import numpy as np
import pandas as pd


def generate(size=100):
    tasks=np.zeros((size, 4), dtype=int)
    tasks[:, 0] = np.arange(0, size, dtype=int)
    tasks[:, 1] = np.random.randint(low=1, high=100, size=size, dtype=int) # reward
    tasks[:, 2] = np.random.randint(low=1, high=100, size=size, dtype=int) # duration
    tasks[:, 3] = np.random.randint(low=1, high=100, size=size, dtype=int) # deadline

    # id = np.arange((1, size+1))
    # reward = np.random.randint(low=1, high=100, size=size, dtype=int)
    # duration = np.random.randint(low=1, high=100, size=size, dtype=int)
    # deadline = np.random.randint(low=1, high=100, size=size, dtype=int)
    
    # for i in range(size):
    #         tasks.append({
    #             'id': i,
    #             'deadline': d[i],
    #             'duration': t[i],
    #             'reward': p[i]
    #         })
    return tasks

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = iterable.tolist()
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1))

def is_overlap():
    pass

def calculate_reward(tasks, ans):
    record = 0
    start_time = 0
    for task_idx in ans:
        # если превысили дедлайн
        if start_time + tasks[task_idx, 2] > tasks[task_idx, 3]:
            return 0
        record += tasks[task_idx, 1]
        start_time += tasks[task_idx, 2]
    return record

def find_optimal(tasks):
    # передадим номера задач и сгенерируем все подмножества 
    # переберем все 2^N вариантов и найдем оптимальное решение
    start_time = 0
    record = 0
    best_solution = None
    # получим подмножество задач
    for ans in powerset(tasks[:, 0]):
        print(ans)
        # проверим, что сгенерированное решение улучшит рекорд
        current_record = calculate_reward(tasks, ans)
        if current_record > record:
            record = current_record
            best_solution = ans

    return best_solution, record

if __name__ == "__main__":
    tasks = generate(5)
    print(tasks)
    solution, record = find_optimal(tasks)
    pd.DataFrame(tasks, columns=["id", "reward", "duration", "deadline"]).to_csv(f"scenario_{record}.csv", index=False)
    