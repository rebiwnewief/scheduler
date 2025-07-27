import sys
import pulp

from main_BnB import TaskScheduler
from generate_scenario import find_optimal


def case_1():
    n = 6
    d = [1, 1, 1, 1, 1, 1]
    t = [1, 1, 1, 1, 1, 1]
    p = [1, 1, 1, 1, 1, 1]
    return n, d, t, p

def case_2():
    n = 3
    d = [5] * n
    t = [3] * n
    p = [1] * n
    return n, d, t, p

def case_3():
    n = 2
    d = [1, 1]
    t = [1, 1]
    p = [10, 5]
    return n, d, t, p

def case_4():
    n = 2
    d = [1, 1]
    t = [2, 2]
    p = [10, 5]
    return n, d, t, p

def case_5():
    n = 2
    d = [1, 1]
    t = [2, 2]
    p = [10, 5]
    return n, d, t, p

def case_test_1():
    n = 4
    d = [6, 1, 3, 7]
    t = [3, 1, 2, 1]
    p = [6, 1, 2, 4]
    return n, d, t, p

def main(n, d, t, profit):
    max_time = max(d)

    prob = pulp.LpProblem('Optimization_problem', pulp.LpMaximize)
    
    # определим переменную: номер задания и время начала
    # в любой момент времени может начаться одна из N задач
    # нужно определить, что задачи, для которых время начала больше делайна
    # не могут существовать
    tasks = pulp.LpVariable.dicts(
        'task',
        ((task, t_start) for task in range(n) for t_start in range(d[task] - t[task] + 1)),
        cat='Binary'
    )
    # время старта + длительность <= дедлайна
    for task in range(n):
        for t_start in range(d[task] - t[task] + 1):
            # если время выполнения меньше дедлайна
            if t_start + t[task] <= d[task]:
                prob += (
                    tasks[task, t_start] * t_start + t[task] <= d[task]
                )
    # каждая задача выполняется не более 1 раза
    for task in range(n):
        prob += pulp.lpSum(
            tasks[task, t_start] for t_start in range(d[task] - t[task] + 1)
        ) <= 1

    # в каждый момент времени выполняется не более 1 задачи
    for t_start in range(0, max_time + 1):
        prob += pulp.lpSum(
            tasks[task, t_start] for task in range(n) if t_start < d[task] - t[task] + 1
        ) <= 1

    # в каждый момент времени проверяем, что интервал 
    # старт текущей задачи, старт текущей задачи + длительность задачи
    # содержит не более двух задач
    for t_start in range(0, max_time + 1):
        for task in range(n):
            if d[task] - t[task] < t_start:
                continue
            for another_task in range(n):
                 if another_task == task:
                     continue
                 for duration in range(1, t[task]):
                    #t_start + duration > d[task] or 
                    if t_start + duration > d[another_task] - t[another_task]:
                        break
                    # попарная невозможные события
                    prob += pulp.lpSum(
                        [tasks[task, t_start], tasks[another_task, t_start + duration]]
                    ) <= 1

    # максимизируем суммарную пользу
    prob += pulp.lpSum(
        tasks[task, t_start] * profit[task] for task in range(n) for t_start in range(d[task] - t[task] + 1)
    )
    solver = pulp.PULP_CBC_CMD(msg=False, warmStart=True)
    prob.solve(solver)

    all_profit = 0
    ans = []    
    for t_start in range(max_time + 1):
        for task in range(n):
            if t_start + t[task] > d[task]:
                continue
            if pulp.value(tasks[task, t_start]) == 1:
                ans.append(task)
                all_profit += profit[task]
    return ans, all_profit

    

if __name__ == "__main__":
    input_filename = 'test.csv'
    # input_filename = 'test_13.csv'
    input_filename = 'test_14.csv'
    
    # чтение из файла
    tasks, record = TaskScheduler.read_tasks_from_csv(input_filename)
    # Найдем оптимальное решение, используя линейное программирование
    optimal_schedule, total_reward = main(
        n =len(tasks),
        d = tasks['deadline'].tolist(),
        t = tasks['duration'].tolist(),
        profit= tasks['reward'].tolist()
        )
    
    print(f"Total reward: {total_reward}")
    print(f"Optimal reward: {record}")
    print("Schedule:")
    if record != 0:
        assert total_reward == record
    start_time = 0
    for _, task in tasks.iloc[optimal_schedule].iterrows():
        print(f"Task {task['id']}: start at {start_time}, finish at {start_time + task['duration']} (reward: {task['reward']})")
        start_time += task['duration']
